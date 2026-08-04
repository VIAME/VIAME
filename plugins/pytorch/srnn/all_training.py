# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import math
import os

import torch

from .utilities import checkpoint, logging


class MemoryBudgetExceeded(RuntimeError):
    """Raised when a stage is close enough to its limit to stop deliberately."""


def memory_limit_gb():
    """What this run is actually allowed, in GB, or None if unbounded.

    Reads the cgroup the job is in, which is what the kernel enforces and
    therefore what kills the process, falling back to what slurm was asked
    for. Nothing here reads the machine's total: a node with 125 GB and a job
    granted 80 is bounded by the 80.
    """
    for path in ( '/sys/fs/cgroup/memory.max',
                  '/sys/fs/cgroup/memory/memory.limit_in_bytes' ):
        try:
            with open( path ) as handle:
                value = handle.read().strip()

            if value and value != 'max':
                limit = int( value ) / ( 1024.0 ** 3 )

                # cgroup v1 reports an enormous number when unlimited
                if limit < 1024 * 1024:
                    return limit
        except ( OSError, ValueError ):
            pass

    for name in ( 'SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU' ):
        value = os.environ.get( name )

        if value and value.isdigit():
            megabytes = int( value )

            if name == 'SLURM_MEM_PER_CPU':
                megabytes *= int( os.environ.get( 'SLURM_CPUS_PER_TASK', 1 ) )

            return megabytes / 1024.0

    return None



class NTMathMixin:
    def __add__(self, other):
        return self._make(e + f for e, f in zip(self, other))

    def __truediv__(self, x):
        return self._make(e / x for e in self)

    @classmethod
    def _zero(cls):
        return cls._make([0] * len(cls._fields))


def normalize_loss(loss):
    if math.isinf(loss):
        logging("WARNING: received an inf loss, setting loss value to 0")
        loss = 0
    return loss


def train_model(
        model, train_loader, test_loader, g_config,
        lr_scheduler, epoch, lr, lr_step, max_iterations,
        run_model, metric_zero, format_metrics, weight_decay=0.0,
):
    # Decay is passed only when a stage asks for it, so a stage that wants
    # none is left at whatever its optimizer's own default is rather than
    # being handed an explicit zero -- the same call it made before this
    # argument existed. The scheduler below rewrites param_groups['lr'] in
    # place and returns the same optimizer, so this survives every step.
    optimizer_args = {'lr': lr}

    if weight_decay:
        optimizer_args['weight_decay'] = weight_decay
        logging('Optimizing with weight decay {}'.format(weight_decay))

    optimizer = g_config.optimizer(model.parameters(), **optimizer_args)

    def run_batch(input_batch, train):
        model.train(train)
        loss, metrics = run_model(input_batch)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return metrics

    limit = memory_limit_gb()
    warned = [False]

    def check_memory(epoch, batch_idx, total_batches):
        if limit is None:
            return

        used, largest, count = process_tree_memory()

        if used is None:
            return

        if used > limit * 0.90:
            logging('Epoch {}: {:.1f} GB of a {:.1f} GB limit at batch {} of '
                    '{}, across {} processes, largest {:.1f} GB. Stopping '
                    'before the kernel does.'
                    .format(epoch, used, limit, batch_idx, total_batches,
                            count, largest))
            raise MemoryBudgetExceeded(
                '{:.1f} GB of a {:.1f} GB limit'.format(used, limit))

        if used > limit * 0.75 and not warned[0]:
            warned[0] = True
            logging('Epoch {}: {:.1f} GB of a {:.1f} GB limit at batch {} of '
                    '{}, across {} processes, largest {:.1f} GB.'
                    .format(epoch, used, limit, batch_idx, total_batches,
                            count, largest))

    def run_epoch(train):
        if train:
            loader = train_loader
            display_interval = g_config.displayInterval
        else:
            loader = test_loader
            display_interval = g_config.vali_displayInterval
        avg_metrics = metric_zero
        final_metrics = metric_zero
        for batch_idx, input_batch in enumerate(loader):
            cur_metrics = run_batch(input_batch, train=train)
            avg_metrics += cur_metrics
            final_metrics += cur_metrics
            # display training info
            if (batch_idx + 1) % display_interval == 0:
                logging(f'Epoch {epoch}: {batch_idx} / {len(loader)} | '
                        + (f'lr:{lr} - t' if train else 'v')
                        + format_metrics(avg_metrics / display_interval))
                avg_metrics = metric_zero

                # Stop on our own terms rather than being killed. A stage that
                # grows without bound is a bug to fix, but until it is found
                # the difference between stopping here and being sent SIGKILL
                # is a snapshot and a reason in the log against neither.
                check_memory(epoch, batch_idx, len(loader))

        # Not meaningful for training since the weights change
        if not train:
            if len(test_loader) == 0:
                return None

            final_metrics /= len(test_loader)
            logging(f'Epoch {epoch}: final v' + format_metrics(final_metrics))
            return final_metrics

    def process_tree_memory():
        """Current RSS of this process and every descendant, in GB.

        Read from /proc rather than from getrusage: RUSAGE_CHILDREN reports
        only children that have been waited for, and with persistent workers
        none have been, so it says almost nothing while the workers are the
        thing growing. This walks the tree instead, and returns the total
        alongside the largest single process, which is what says whether the
        growth is in the parent or spread across the loaders.
        """
        try:
            import os as _os

            def rss_kb(pid):
                # Pss, not Rss. Workers share most of their mapping with the
                # parent, so each reports the same pages and summing Rss over
                # the tree counts them once per process: five processes
                # sharing 57 GB came out as 170 GB, which tripped the budget
                # guard against a limit the run was nowhere near. Pss divides
                # a shared page between those sharing it, so the sum over a
                # tree is what the tree actually occupies.
                try:
                    with open('/proc/{}/smaps_rollup'.format(pid)) as handle:
                        for line in handle:
                            if line.startswith('Pss:'):
                                return int(line.split()[1])
                except OSError:
                    pass

                with open('/proc/{}/statm'.format(pid)) as handle:
                    return int(handle.read().split()[1]) * (
                        _os.sysconf('SC_PAGE_SIZE') // 1024)

            def children(pid):
                try:
                    with open('/proc/{}/task/{}/children'.format(pid, pid)) as h:
                        return [int(x) for x in h.read().split()]
                except OSError:
                    return []

            pids, stack = [], [_os.getpid()]

            while stack:
                pid = stack.pop()
                pids.append(pid)
                stack.extend(children(pid))

            sizes = []

            for pid in pids:
                try:
                    sizes.append(rss_kb(pid))
                except OSError:
                    pass

            if not sizes:
                return None, None, 0

            return (sum(sizes) / 1048576.0, max(sizes) / 1048576.0,
                    len(sizes))
        except Exception:
            return None, None, 0

    def resident_memory():
        """This process and its children, in GB, or None where unavailable.

        Reported each epoch because these stages have twice been killed by a
        signal with nothing in the log to say why, and a kill from outside
        looks identical whether it was memory or anything else. A number here
        settles it from the log alone rather than from a rerun.
        """
        try:
            import resource

            total = resource.getrusage( resource.RUSAGE_SELF ).ru_maxrss
            total += resource.getrusage( resource.RUSAGE_CHILDREN ).ru_maxrss

            return total / ( 1024.0 * 1024.0 )  # ru_maxrss is KB on linux
        except Exception:
            return None

    # Validation-loss patience. Selection afterwards picks the best epoch
    # from the record, so epochs past the plateau only cost wall clock; the
    # first metric field is the loss for every stage that comes through here.
    patience = getattr(g_config, 'early_stop_patience', None)
    best_validation = None
    epochs_since_best = 0

    # train loop
    for epoch in range(epoch, max_iterations):
        # change learning rate
        optimizer, lr = lr_scheduler(optimizer, epoch, lr, lr_step)

        peak = resident_memory()
        total, largest, count = process_tree_memory()

        if total is not None:
            logging( 'Epoch {}: memory {:.1f} GB across {} processes, '
                     'largest {:.1f} GB (peak so far {:.1f} GB)'
                     .format( epoch, total, count, largest,
                              peak if peak is not None else 0.0 ) )
        elif peak is not None:
            logging( 'Epoch {}: peak resident memory {:.1f} GB'
                     .format( epoch, peak ) )

        try:
            run_epoch(train=True)
            validation_metrics = run_epoch(train=False)
        except MemoryBudgetExceeded as exceeded:
            # Keep what this epoch reached before stopping. Being killed loses
            # the epoch outright; this way the stage carries on from here once
            # the run is given more memory or the growth is fixed.
            partial = os.path.join(
                g_config.model_dir, 'snapshot_epoch_{}.pt'.format(epoch))
            torch.save(checkpoint(model, epoch), partial)
            logging('Snapshot saved to {} before stopping'.format(partial))
            logging('Stopped inside epoch {}: {}. Resume to carry on from '
                    'this snapshot.'.format(epoch, exceeded))
            raise

        # save snapshot
        save_path = os.path.join(g_config.model_dir, 'snapshot_epoch_{}.pt'.format(epoch))
        torch.save(checkpoint(model, epoch), save_path)
        logging('Snapshot saved to {}'.format(save_path))

        if patience and validation_metrics is not None:
            validation_loss = validation_metrics[0]

            if best_validation is None \
                    or validation_loss < best_validation - 1e-5:
                best_validation = validation_loss
                epochs_since_best = 0
            else:
                epochs_since_best += 1

                if epochs_since_best >= patience:
                    logging('No validation improvement for {} epochs '
                            '(best {:.5f}); stopping at epoch {} of {}.'
                            .format(patience, best_validation, epoch,
                                    max_iterations))
                    break

    else:
        # terminate
        logging('Maximum epoch reached, terminating ...')
