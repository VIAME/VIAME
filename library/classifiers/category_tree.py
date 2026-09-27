# Torch CategoryTree operations vendored from ndsampler 0.8.1.
# Kept separate from its image sampler so inference does not import demos.
#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/
# 
#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
# 
#    1. Definitions.
# 
#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.
# 
#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.
# 
#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.
# 
#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.
# 
#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.
# 
#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.
# 
#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).
# 
#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.
# 
#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."
# 
#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.
# 
#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.
# 
#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.
# 
#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:
# 
#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and
# 
#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and
# 
#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and
# 
#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.
# 
#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.
# 
#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.
# 
#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.
# 
#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.
# 
#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.
# 
#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.
# 
#    END OF TERMS AND CONDITIONS
# 
#    APPENDIX: How to apply the Apache License to your work.
# 
#       To apply the Apache License to your work, attach the following
#       boilerplate notice, with the fields enclosed by brackets "{}"
#       replaced with your own identifying information. (Don't include
#       the brackets!)  The text should be enclosed in the appropriate
#       comment syntax for the file format. We also recommend that a
#       file or class name and description of purpose be included on the
#       same "printed page" as the copyright notice for easier
#       identification within third-party archives.
# 
#    Copyright 2022 "Kitware Inc"
# 
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
# 
#        http://www.apache.org/licenses/LICENSE-2.0
# 
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
Extends the :class:`CategoryTree` class in the :mod:`kwcoco.category_tree`
module with torch methods for computing hierarchical losses / decisions.


Notes from YOLO-9000:
    * perform multiple softmax operations over co-hyponyms
    * we compute the softmax over all sysnsets that are hyponyms of the same concept

    synsets - sets of synonyms (word or phrase that means exactly or nearly the same as another)

    hyponymn - a word of more specific meaning than a general or
        superordinate term applicable to it. For example, spoon is a
        hyponym of cutlery.
"""
import kwarray
import functools
import networkx as nx
import ubelt as ub
# import torch
# import torch.nn.functional as F
import numpy as np
from kwcoco import CategoryTree as KWCOCO_CategoryTree  # raw category tree

__all__ = ['CategoryTree']


class Mixin_CategoryTree_Torch:
    """
    Mixin methods for CategoryTree that specifically relate to computing
    normalized probabilities.
    """

    def conditional_log_softmax(self, class_energy, dim):
        """
        Computes conditional log probabilities of each class in the category tree

        Args:
            class_energy (Tensor): raw values output by final network layer
            dim (int): dimension where each index corresponds to a class

        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> import torch
            >>> from viame.classifiers.category_tree import *
            >>> graph = nx.generators.gnr_graph(30, 0.3, seed=321).reverse()
            >>> self = CategoryTree(graph)
            >>> class_energy = torch.randn(64, len(self.idx_to_node))
            >>> cond_logprobs = self.conditional_log_softmax(class_energy, dim=1)
            >>> # The sum of the conditional probabilities should be
            >>> # equal to the number of sibling groups.
            >>> cond_probs = torch.exp(cond_logprobs).numpy()
            >>> assert np.allclose(cond_probs.sum(axis=1), len(self.idx_groups))
        """
        import torch
        import torch.nn.functional as F
        cond_logprobs = torch.empty_like(class_energy)
        if class_energy.numel() == 0:
            return cond_logprobs
        # Move indexes onto the class_energy device (perhaps precache this)
        index_groups = [torch.LongTensor(idxs).to(class_energy.device)
                        for idxs in self.idx_groups]
        # Note: benchmarks for implementation choices are in ndsampler/dev
        # The index_select/index_copy_ solution is faster than fancy indexing
        for index in index_groups:
            # Take each subset of classes that are mutually exclusive
            energy_group = torch.index_select(class_energy, dim=dim, index=index)
            # Then apply the log_softmax to those sets
            logprob_group = F.log_softmax(energy_group, dim=dim)
            cond_logprobs.index_copy_(dim, index, logprob_group)
        return cond_logprobs

    def _apply_logit_chain_rule(self, cond_logprobs, dim):
        """
        Incorrect name for backwards compatibility
        """
        import warnings
        warnings.warn(
            'Use _apply_logit_chain_rule is incorrectly named '
            'and deprecated use _apply_logprob_chain_rule instead',
            DeprecationWarning)
        return self._apply_logprob_chain_rule(cond_logprobs, dim)

    def _apply_logprob_chain_rule(self, cond_logprobs, dim):
        """
        Applies the probability chain rule (in log space, which has better
        numerical properties) to a set of conditional probabilities (wrt this
        hierarchy) to achieve absolute probabilities for each node.

        Args:
            cond_logprobs (Tensor): conditional log probabilities for each class
            dim (int): dimension where each index corresponds to a class

        Notes:
            Probability chain rule:
                P(node) = P(node | parent) * P(parent)

            Log-Probability chain rule:
                log(P(node)) = log(P(node | parent)) + log(P(parent))
        """
        import torch
        # The dynamic program was faster on the CPU in a dummy test case
        memo = {}

        def log_prob(node, memo=memo):
            """ dynamic program to compute absolute class log probability """
            if node in memo:
                return memo[node]
            logp_node_given_parent = cond_logprobs.select(dim, self.node_to_idx[node])
            parents = list(self.graph.predecessors(node))
            if len(parents) == 0:
                logp_node = logp_node_given_parent
            elif len(parents) == 1:
                # log(P(node)) = log(P(node | parent)) + log(P(parent))
                logp_node = logp_node_given_parent + log_prob(parents[0])
            else:
                raise AssertionError('not a tree')
            memo[node] = logp_node
            return logp_node

        class_logprobs = torch.empty_like(cond_logprobs)
        if cond_logprobs.numel() > 0:
            for idx, node in enumerate(self.idx_to_node):
                # Note: the this is the bottleneck in this function
                if True:
                    class_logprobs.select(dim, idx)[:] = log_prob(node)
                else:
                    result = log_prob(node)  # 50% of the time
                    _dest = class_logprobs.select(dim, idx)  # 8% of the time
                    _dest[:] = result  # 37% of the time
        return class_logprobs

    def source_log_softmax(self, class_energy, dim):
        """
        Top-down hierarchical softmax. This is the default
        hierarchical_log_softmax function.

        Alternative to `sink_log_softmax`

        SeeAlso:
            * sink_log_softmax
            * hierarchical_log_softmax (alias for this function)
            * source_log_softmax (this function)

        Converts raw class energy to absolute log probabilities based on the
        category hierarchy. This is done by first converting to conditional
        log probabilities and then applying the probability chain rule (in log
        space, which has better numerical properties).

        Args:
            class_energy (Tensor): raw output from network. The values in
                `class_energy[..., idx]` should correspond to the network
                activations for the hierarchical class `self.idx_to_node[idx]`
            dim (int): dimension corresponding to classes (usually 1)

        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> import torch
            >>> from viame.classifiers.category_tree import *
            >>> graph = nx.generators.gnr_graph(20, 0.3, seed=328).reverse()
            >>> self = CategoryTree(graph)
            >>> class_energy = torch.randn(3, len(self.idx_to_node))
            >>> class_logprobs = self.hierarchical_log_softmax(class_energy, dim=-1)
            >>> class_probs = torch.exp(class_logprobs)
            >>> for node, idx in self.node_to_idx.items():
            ...     # Check the total children probabilities
            ...     # is equal to the parent probability.
            ...     children = list(self.graph.successors(node))
            ...     child_idxs = [self.node_to_idx[c] for c in children]
            ...     if len(child_idxs) > 0:
            ...         child_sum = class_probs[..., child_idxs].sum(dim=1)
            ...         p_node = class_probs[..., idx]
            ...         torch.allclose(child_sum, p_node)
        """
        if dim < 0:
            dim = dim % class_energy.ndimension()
        cond_logprobs = self.conditional_log_softmax(class_energy, dim=dim)
        class_logprobs = self._apply_logprob_chain_rule(cond_logprobs, dim=dim)
        return class_logprobs

    def sink_log_softmax(self, class_energy, dim):
        """
        Bottom-up hierarchical softmax.

        Alternative to `source_log_softmax`

        SeeAlso:
            * sink_log_softmax (this function)
            * hierarchical_log_softmax (alias for source log softmax)
            * source_log_softmax

        Does a regular softmax over all the mutually exclusive leaf nodes, then
        sums their probabilities to get the score for the parent nodes.

        Notes:
            In this method of computation ONLY the energy in the leaf nodes
            matters. The energy in all other intermediate notes is ignored.
            For this reason the "source" method of computation is often
            preferred.

        Args:
            class_energy (Tensor): raw output from network. The values in
                `class_energy[..., idx]` should correspond to the network
                activations for the hierarchical class `self.idx_to_node[idx]`
            dim (int): dimension corresponding to classes (usually 1)

        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> import torch
            >>> from viame.classifiers.category_tree import *
            >>> graph = nx.generators.gnr_graph(20, 0.3, seed=328).reverse()
            >>> self = CategoryTree(graph)
            >>> class_energy = torch.randn(3, len(self.idx_to_node))
            >>> class_logprobs = self.hierarchical_log_softmax(class_energy, dim=1)
            >>> class_probs = torch.exp(class_logprobs)
            >>> for node, idx in self.node_to_idx.items():
            ...     # Check the total children probabilities
            ...     # is equal to the parent probability.
            ...     children = list(self.graph.successors(node))
            ...     child_idxs = [self.node_to_idx[c] for c in children]
            ...     if len(child_idxs) > 0:
            ...         child_sum = class_probs[..., child_idxs].sum(dim=1)
            ...         p_node = class_probs[..., idx]
            ...         torch.allclose(child_sum, p_node)

        Ignore:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> import torch
            >>> class_logprobs1 = self.sink_log_softmax(class_energy, dim=1)
            >>> class_logprobs2 = self.source_log_softmax(class_energy, dim=1)
            >>> class_probs1 = torch.exp(class_logprobs1)
            >>> class_probs2 = torch.exp(class_logprobs2)
        """
        import torch
        import torch.nn.functional as F
        class_logprobs = torch.empty_like(class_energy)
        leaf_idxs = sorted(self.node_to_idx[node]
                           for node in sink_nodes(self.graph))
        leaf_idxs = torch.LongTensor(leaf_idxs).to(class_energy.device)

        leaf_energy = torch.index_select(class_energy, dim=dim,
                                         index=leaf_idxs)
        leaf_logprobs = F.log_softmax(leaf_energy, dim=dim)
        class_logprobs.index_copy_(dim, leaf_idxs, leaf_logprobs)

        @ub.memoize
        def populate2(node):
            """ dynamic program to compute absolute class log probability """
            children = list(self.graph.successors(node))
            if len(children) > 0:
                # Ensure that all children are populated before the parents
                for child in children:
                    populate2(child)
                child_idxs = sorted(self.node_to_idx[node] for node in children)
                child_idxs = torch.LongTensor(child_idxs).to(class_energy.device)
                node_idx = self.node_to_idx[node]
                selected = torch.index_select(class_logprobs, dim=dim,
                                              index=child_idxs)
                total = torch.logsumexp(selected, dim=dim)  # sum in logspace
                class_logprobs.select(dim, node_idx)[:] = total

        for node in self.graph.nodes():
            populate2(node)
        return class_logprobs

    # TODO: Need to figure out the best way to parametarize which
    # of the source or sink softmaxes will be used by a network
    hierarchical_log_softmax = source_log_softmax
    # hierarchical_log_softmax = sink_log_softmax

    def hierarchical_softmax(self, class_energy, dim):
        """ Convenience method which converts class-energy to final probs """
        import torch
        class_logprobs = self.hierarchical_log_softmax(class_energy, dim)
        class_probs = torch.exp(class_logprobs)
        return class_probs

    def graph_log_softmax(self, class_energy, dim):
        """ Convenience method which converts class-energy to logprobs """
        class_logprobs = self.hierarchical_log_softmax(class_energy, dim)
        return class_logprobs

    def graph_softmax(self, class_energy, dim):
        """ Convenience method which converts class-energy to final probs """
        import torch
        class_logprobs = self.hierarchical_log_softmax(class_energy, dim)
        class_probs = torch.exp(class_logprobs)
        return class_probs

    def hierarchical_cross_entropy(self, class_energy, targets,
                                   reduction='mean'):
        """
        Combines hierarchical_log_softmax and nll_loss in a single function
        """
        import torch.nn.functional as F
        class_logprobs = self.hierarchical_log_softmax(class_energy, dim=1)
        loss = F.nll_loss(class_logprobs, targets, reduction=reduction)
        return loss

    def hierarchical_nll_loss(self, class_logprobs, targets):
        """
        Given predicted hierarchical class log-probabilities and target vectors
        indicating the finest-grained known target class, compute the loss such
        that only errors on coarser levels of the hierarchy are considered.
        To quote from YOLO-9000:
            > For example, if the label is “dog” we do assign any error to
            > predictions further down in the tree, “German Shepherd” versus
            > “Golden Retriever”, because we do not have that information.

        Note that all the hard word needed to consider all coarser classes and
        ignore all finer grained classes is done in the computation of
        `class_logprobs`. Given these and targets specified at the finest known
        category for each target (which might be very coarse), we can simply
        use regular nll_loss and everything will work out.

        This is because the class logprobs have already been computed using
        information from all conditional probabilities computed at itself and
        at each ancestor in the tree, and these conditional probabilities were
        computed with respect to all siblings at a given level, so the class
        logprobs exactly contain the relevant information. In other words as long
        as the log probabilities are computed correctly, then the nothing
        special needs to happen when computing the loss.

        Args:
            class_logprobs (Tensor): log probabilities for each class
            targets (Tensor): true class for each example

        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> from viame.classifiers.category_tree import *  # NOQA
            >>> import networkx as nx
            >>> import torch
            >>> graph = nx.from_dict_of_lists({
            >>>     'background': [],
            >>>     'mineral': ['granite', 'quartz'],
            >>>     'animal': ['dog', 'cat'],
            >>>     'dog': ['boxer', 'beagle']
            >>> }, nx.DiGraph)
            >>> self = CategoryTree(graph)
            >>> target_nodes = ['boxer', 'dog', 'quartz', 'animal', 'background']
            >>> targets = torch.LongTensor([self.node_to_idx[n] for n in target_nodes])
            >>> class_energy = torch.randn(len(targets), len(self), requires_grad=True)
            >>> class_logprobs = self.hierarchical_log_softmax(class_energy, dim=-1)
            >>> loss = self.hierarchical_nll_loss(class_logprobs, targets)
            >>> # Note that only the relevant classes (wrt to the target) for
            >>> # each batch item receive gradients
            >>> loss.backward()
            >>> is_relevant = (class_energy.grad.abs() > 0).numpy()
            >>> print('target -> relevant')
            >>> print('------------------')
            >>> for bx, idx in enumerate(sorted(targets.tolist())):
            ...     target_node = self.idx_to_node[idx]
            ...     relevant_nodes = sorted(ub.take(self.idx_to_node, np.where(is_relevant[bx])[0]))
            ...     print('{} -> {}'.format(target_node, relevant_nodes))
            target -> relevant
            ------------------
            background -> ['animal', 'background', 'beagle', 'boxer', 'cat', 'dog', 'mineral']
            animal -> ['animal', 'background', 'cat', 'dog', 'mineral']
            dog -> ['animal', 'background', 'granite', 'mineral', 'quartz']
            quartz -> ['animal', 'background', 'mineral']
            boxer -> ['animal', 'background', 'mineral']
        """
        import torch.nn.functional as F
        loss = F.nll_loss(class_logprobs, targets)
        return loss

    def _prob_decision(self, class_probs, dim, thresh=0.1):
        """
        Chooses the finest-grained category based on raw prob threshold

        Args:
            thresh (float): only make a more fine-grained decision if
                its probability is above this threshold. This number should
                be set relatively low to encourage smoothness in the
                detection metrics.

        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> import torch
            >>> from viame.classifiers.category_tree import *
            >>> import torch
            >>> from viame.classifiers import category_tree
            >>> self = category_tree.CategoryTree.demo('btree', r=3, h=4)
            >>> class_energy = torch.randn(33, len(self))
            >>> class_logprobs = self.hierarchical_log_softmax(class_energy, dim=-1)
            >>> class_probs = torch.exp(class_logprobs).numpy()
            >>> dim = 1
            >>> thresh = 0.3
            >>> pred_idxs, pred_conf = self.decision(class_probs, dim, thresh=thresh)
            >>> pred_cnames = list(ub.take(self.idx_to_node, pred_idxs))
        """
        import kwarray
        impl = kwarray.ArrayAPI.impl(class_probs)

        sources = list(source_nodes(self.graph))
        other_dims = sorted(set(range(len(class_probs.shape))) - {dim})

        # Rearange probs so the class dimension is at the end
        # flat_class_probs = class_probs.transpose(*other_dims + [dim]).reshape(-1, class_probs.shape[dim])
        flat_class_probs = impl.transpose(class_probs, other_dims + [dim]).reshape(-1, class_probs.shape[dim])
        flat_jdxs = np.arange(flat_class_probs.shape[0])

        def _descend(depth, nodes, jdxs):
            """
            Recursively descend the class tree starting at the coarsest level.
            At each level we decide if the items will take a category at this
            level of granularity or try to take a more fine-grained label.

            Args:
                depth (int): current depth in the tree
                nodes (list) : set of sibling nodes at a this level
                jdxs (ArrayLike): item indices that made it to this level (note
                    idxs are used for class indices)
            """
            # Look at the probabilities of each node at this level
            idxs = sorted(self.node_to_idx[node] for node in nodes)
            probs = flat_class_probs[jdxs][:, idxs]

            pred_conf, pred_cx = impl.max_argmax(probs, axis=1)
            pred_idxs = np.array(idxs)[pred_cx]

            # Keep desending on items above the threshold
            # TODO: is there a more intelligent way to do this?
            check_children = pred_conf > thresh

            if impl.any(check_children):
                # Check the children of these nodes
                check_jdxs = jdxs[check_children]
                check_idxs = pred_idxs[check_children]
                group_idxs, groupxs = kwarray.group_indices(check_idxs)
                for idx, groupx in zip(group_idxs, groupxs):
                    node = self.idx_to_node[idx]
                    children = list(self.graph.successors(node))
                    if children:
                        sub_jdxs = check_jdxs[groupx]
                        # See if any fine-grained categories also have high
                        # thresholds.
                        sub_idxs, sub_conf = _descend(depth + 1, children,
                                                      sub_jdxs)
                        sub_flags = sub_conf > thresh
                        # Overwrite course decisions with confident
                        # fine-grained ones.
                        fine_groupx = groupx[sub_flags]
                        fine_idxs = sub_idxs[sub_flags]
                        fine_conf = sub_conf[sub_flags]
                        pred_conf[fine_groupx] = fine_conf
                        pred_idxs[fine_groupx] = fine_idxs
            return pred_idxs, pred_conf

        nodes = sources
        jdxs = flat_jdxs
        pred_idxs, pred_conf = _descend(0, nodes, jdxs)
        return pred_idxs, pred_conf

    def _demo_probs(self, num=5, rng=0, nonrandom=3, hackargmax=True):
        """ dummy probabilities for testing """
        import torch
        rng = kwarray.ensure_rng(rng)
        class_energy = torch.FloatTensor(rng.rand(num, len(self)))

        # Setup the first few examples to prefer being classified
        # as a fine grained class to a decreasing degree.
        # The first example is set to have equal energy
        # The i + 2-th example is set to have an extremely high energy.
        start = 0
        nonrandom = min(nonrandom, (num - start))
        if nonrandom > 0:
            path = sorted(ub.take(self.node_to_idx, nx.dag_longest_path(self.graph)))

            class_energy[start] = 1 / len(class_energy[start])
            if hackargmax:
                # HACK: even though we want to test uniform distributions, it makes
                # regression tests difficiult because torch and numpy return a
                # different argmax when the array has more than one max value.
                # add a VERY small epsilon to make max values distinct
                class_energy[start] += torch.linspace(0, .00001, len(class_energy[start]))

            if nonrandom > 1:
                for i in range(nonrandom - 2):
                    class_energy[start + i + 1][path] += 2 ** (i / 4)
                class_energy[start + i + 2][path] += 2 ** 20

        class_probs = self.hierarchical_softmax(class_energy, dim=1)
        return class_probs

    def decision(self, class_probs, dim, thresh=0.5, criterion='gini',
                 ignore_class_idxs=None, always_refine_idxs=None):
        """
        Chooses the finest-grained category based on information gain

        Args:
            thresh (float): threshold on simplicity ratio.
                Small thresholds are more permissive, i.e. the returned classes
                will often be more fined-grained. Larger thresholds are less
                permissive and prefer coarse-grained classes.

            criterion (str): how to compute information. Either entropy or gini.

            ignore_class_idxs (List[int], optional): if specified this is a list
                of class indices which we are not allowed to predict. We
                will proceed as if the graph did not contain these nodes.
                (Useful for getting low-probability detections).

            always_refine_idxs  (List[int], optional):
                if specified this is a list of class indices that we will
                always refine into a more fine-grained class.
                (Useful if you have a dummy root)

        Returns:
            Tuple[Tensor, Tensor]: pred_idxs, pred_conf:
                pred_idxs: predicted class indices
                pred_conf: associated confidence

        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> import torch
            >>> from viame.classifiers.category_tree import *
            >>> self = CategoryTree.demo('btree', r=3, h=3)
            >>> rng = kwarray.ensure_rng(0)
            >>> class_energy = torch.FloatTensor(rng.rand(33, len(self)))
            >>> # Setup the first few examples to prefer being classified
            >>> # as a fine grained class to a decreasing degree.
            >>> # The first example is set to have equal energy
            >>> # The i + 2-th example is set to have an extremely high energy.
            >>> path = sorted(ub.take(self.node_to_idx, nx.dag_longest_path(self.graph)))
            >>> class_energy[0] = 1 / len(class_energy[0])
            >>> for i in range(8):
            >>>     class_energy[i + 1][path] += 2 ** (i / 4)
            >>> class_energy[i + 2][path] += 2 ** 20
            >>> class_logprobs = self.hierarchical_log_softmax(class_energy, dim=-1)
            >>> class_probs = torch.exp(class_logprobs).numpy()
            >>> print(ub.hzcat(['probs = ', ub.urepr(class_probs[:8], precision=2, supress_small=True)]))
            >>> dim = 1
            >>> criterion = 'entropy'
            >>> thresh = 0.40
            >>> pred_idxs, pred_conf = self.decision(class_probs[0:10], dim, thresh=thresh, criterion=criterion)
            >>> print('pred_conf = {!r}'.format(pred_conf))
            >>> print('pred_idxs = {!r}'.format(pred_idxs))
            >>> pred_cnames = list(ub.take(self.idx_to_node, pred_idxs))

        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> import torch
            >>> from viame.classifiers.category_tree import *
            >>> self = CategoryTree.demo('btree', r=3, h=3)
            >>> class_probs = self._demo_probs()
            >>> # Test ignore_class_idxs
            >>> self.decision(class_probs, dim=1, ignore_class_idxs=[0])
            >>> # Should not be able to ignore all top level nodes
            >>> import pytest
            >>> with pytest.raises(ValueError):
            >>>     self.decision(class_probs, dim=1, ignore_class_idxs=self.idx_groups[0])
            >>> # But it is OK to ignore all child nodes at a particular level
            >>> self.decision(class_probs, dim=1, ignore_class_idxs=self.idx_groups[1])

        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> import torch
            >>> from viame.classifiers.category_tree import *
            >>> self = CategoryTree.demo('btree', r=3, h=3, add_zero=False)
            >>> class_probs = self._demo_probs(num=30, nonrandom=20)
            >>> pred_idxs0, pref_conf0 = self.decision(class_probs, dim=1, always_refine_idxs=[])
            >>> assert 0 in pred_idxs0
            >>> ###
            >>> #print(ub.color_text('!!!!!!!!!!!!!!!!!!!', 'white'))
            >>> pred_idxs1, pref_conf1 = self.decision(class_probs, dim=1, always_refine_idxs=[0])
            >>> #print(ub.color_text('!!!!!!!!!!!!!!!!!!!', 'red'))
            >>> pred_idxs2, pref_conf2 = self.decision(class_probs.numpy(), dim=1, always_refine_idxs=[0])
            >>> assert np.all(pred_idxs1 == pred_idxs2)
            >>> assert 0 not in pred_idxs1

        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> import torch
            >>> from viame.classifiers.category_tree import *
            >>> graph = nx.from_dict_of_lists({
            >>>     'a': ['b', 'q'],
            >>>     'b': ['c'],
            >>>     'c': ['d'],
            >>>     'd': ['e', 'f'],
            >>> }, nx.DiGraph)
            >>> self = CategoryTree(graph)
            >>> class_probs = self._demo_probs()
            >>> pred_idxs1, pref_conf1 = self.decision(class_probs, dim=1)
            >>> self = CategoryTree.demo('btree', r=1, h=4, add_zero=False)
            >>> class_probs = self._demo_probs()
            >>> # We should always descend to the finest level if we just have a straight line
            >>> pred_idxs1, pref_conf1 = self.decision(class_probs, dim=1)
            >>> assert np.all(pred_idxs1 == 4)

        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> import torch
            >>> # FIXME: What do we do in this case?
            >>> # Do we always descend at level A?
            >>> from viame.classifiers.category_tree import *
            >>> graph = nx.from_dict_of_lists({
            >>>     'a': ['b', 'c'],
            >>> }, nx.DiGraph)
            >>> self = CategoryTree(graph)
            >>> class_probs = self._demo_probs(num=10, nonrandom=8)
            >>> pred_idxs1, pref_conf1 = self.decision(class_probs, dim=1)
            >>> print('pred_idxs1 = {!r}'.format(pred_idxs1))

        Ignore:
            >>> import kwcoco
            >>> from kwcoco.category_tree import _print_forest
            >>> dset = kwcoco.CocoDataset.demo('shapes8')
            >>> self = dset.object_categories()
            >>> _print_forest(self.graph)
            ├── background
            ├── raster
            │   ├── eff
            │   └── superstar
            └── vector
                └── star

            >>> print(list(self))
            ['background', 'star', 'superstar', 'eff', 'raster', 'vector']

            >>> class_probs = np.array([[0.05, 0.10, 0.90, 0.10, .90, .05]])
            >>> self.decision(class_probs, dim=1)
            >>> print('node = {!r}, prob = {!r}'.format(self.idx_to_node[idx[0]], prob[0]))

            >>> class_probs = np.array([[0.00, 0.00, 0.98, 0.02, 0.98, 0.02]])
            >>> idx, prob = self.decision(class_probs, dim=1, thresh=0.1, criterion='entropy')
            >>> print('node = {!r}, prob = {!r}'.format(self.idx_to_node[idx[0]], prob[0]))

            >>> class_probs = np.array([[0.00, 0.98, 0.01, 0.01, 0.02, 0.98]])
            >>> idx, prob = self.decision(class_probs, dim=1, thresh=0.01, criterion='entropy')
            >>> print('node = {!r}, prob = {!r}'.format(self.idx_to_node[idx[0]], prob[0]))

            >>> class_probs = np.array([[0.00, 1.00, 0.00, 0.00, 0.00, 1.00]])
            >>> idx, prob = self.decision(class_probs, dim=1, thresh=0.1, criterion='entropy')
            >>> print('node = {!r}, prob = {!r}'.format(self.idx_to_node[idx[0]], prob[0]))
        """
        if criterion == 'prob':
            return self._prob_decision(class_probs, dim, thresh=thresh)

        DEBUG = False

        impl = kwarray.ArrayAPI.impl(class_probs)

        sources = list(source_nodes(self.graph))
        other_dims = sorted(set(range(len(class_probs.shape))) - {dim})

        # Rearrange probs so the class dimension is at the end
        flat_class_probs = impl.transpose(class_probs, other_dims + [dim]).reshape(-1, class_probs.shape[dim])
        flat_jdxs = np.arange(flat_class_probs.shape[0])

        if criterion == 'gini':
            _criterion = functools.partial(gini, axis=dim, impl=impl)
            # _criterion = functools.partial(gini, axis=dim)
        elif criterion == 'entropy':
            _criterion = functools.partial(entropy, axis=dim, impl=impl)
            # _criterion = functools.partial(entropy, axis=dim)
        else:
            raise KeyError(criterion)

        def _entropy_refine(depth, nodes, jdxs):
            """
            Recursively descend the class tree starting at the coursest level.
            At each level we decide if the items will take a category at this
            level of granulatority or try to take a more fine-grained label.

            Args:
                depth (int): current depth in the tree
                nodes (list) : set of sibling nodes at a this level
                jdxs (ArrayLike): item indices that made it to this level (note
                    idxs are used for class indices)
            """
            if DEBUG:
                print(ub.color_text('* REFINE nodes={}'.format(nodes), 'blue'))
            # Look at the probabilities of each node at this level
            idxs = sorted(self.node_to_idx[node] for node in nodes)
            if ignore_class_idxs:
                ignore_nodes = set(ub.take(self.idx_to_node, ignore_class_idxs))
                idxs = sorted(set(idxs) - set(ignore_class_idxs))
                if len(idxs) == 0:
                    raise ValueError('Cannot ignore all top-level classes')
            probs = flat_class_probs[jdxs][:, idxs]

            # Choose a highest probability category to predict at this level
            pred_conf, pred_cx = impl.max_argmax(probs, axis=1)
            pred_idxs = np.array(idxs)[impl.numpy(pred_cx)]

            # Group each example which predicted the same class at this level
            group_idxs, groupxs = kwarray.group_indices(pred_idxs)
            if DEBUG:
                groupxs = list(ub.take(groupxs, group_idxs.argsort()))
                group_idxs = group_idxs[group_idxs.argsort()]
                # print('groupxs = {!r}'.format(groupxs))
                # print('group_idxs = {!r}'.format(group_idxs))

            for idx, groupx in zip(group_idxs, groupxs):
                # Get the children of this node (idx)
                node = self.idx_to_node[idx]
                children = sorted(self.graph.successors(node))
                if ignore_class_idxs:
                    children = sorted(set(children) - ignore_nodes)

                if children:
                    # Check if it would be simple to refine the coarse category
                    # current prediction into one of its finer-grained child
                    # categories. Do this by considering the entropy at this
                    # level if we replace this coarse-node with the child
                    # fine-nodes. Then compare that entropy to what we would
                    # get if we were perfectly uncertain about the child node
                    # prediction (i.e. the worst case). If the entropy we get
                    # is much lower than the worst case, then it is simple to
                    # descend the tree and predict a finer-grained label.

                    # Expand this node into all of its children
                    child_idxs = set(self.node_to_idx[child] for child in children)

                    # Get example indices (jdxs) assigned to category idx
                    groupx.sort()
                    group_jdxs = jdxs[groupx]

                    # Expand this parent node, but keep the parent's siblings
                    ommer_idxs = sorted(set(idxs) - {idx})  # Note: ommer = Aunt/Uncle
                    expanded_idxs = sorted(ommer_idxs) + sorted(child_idxs)
                    expanded_probs = flat_class_probs[group_jdxs][:, expanded_idxs]

                    # Compute the entropy of the expanded distribution
                    h_expanded = _criterion(expanded_probs)

                    # Probability assigned to the parent
                    p_parent = flat_class_probs[group_jdxs][:, idx:idx + 1]
                    # Get the absolute probabilities assigned the parents siblings
                    ommer_probs = flat_class_probs[group_jdxs][:, sorted(ommer_idxs)]

                    # Compute the worst-case entropy after expanding the node
                    # In the worst case the parent probability is distributed
                    # uniformly among all of its children
                    c = len(children)
                    child_probs_worst = impl.tile(p_parent / c, reps=[1, c])
                    expanded_probs_worst = impl.hstack([ommer_probs, child_probs_worst])
                    h_expanded_worst = _criterion(expanded_probs_worst)

                    # Normalize the entropy we got by the worst case.
                    # eps = float(np.finfo(np.float32).min)
                    eps = 1e-30
                    complexity_ratio = h_expanded / (h_expanded_worst + eps)
                    simplicity_ratio = 1 - complexity_ratio

                    # If simplicity ratio is over a threshold refine the parent
                    refine_flags = simplicity_ratio > thresh

                    if always_refine_idxs is not None:
                        if idx in always_refine_idxs:
                            refine_flags[:] = 1

                    if len(child_idxs) == 1:
                        # hack: always refine when there is one child, in this
                        # case the simplicity measure will always be zero,
                        # which is likely a problem with this criterion.
                        refine_flags[:] = 1

                    refine_flags = kwarray.ArrayAPI.numpy(refine_flags).astype(bool)

                    if DEBUG:
                        print('-----------')
                        print('idx = {!r}'.format(idx))
                        print('node = {!r}'.format(self.idx_to_node[idx]))
                        print('ommer_idxs = {!r}'.format(ommer_idxs))
                        print('ommer_nodes = {!r}'.format(
                            list(ub.take(self.idx_to_node, ommer_idxs))))
                        print('depth = {!r}'.format(depth))
                        import pandas as pd
                        print('expanded_probs =\n{}'.format(
                            ub.urepr(expanded_probs, precision=2,
                                     with_dtype=0, supress_small=True)))
                        df = pd.DataFrame({
                            'h': h_expanded,
                            'h_worst': h_expanded_worst,
                            'c_ratio': complexity_ratio,
                            's_ratio': simplicity_ratio,
                            'flags': refine_flags.astype(np.uint8)
                        })
                        print(df)
                        print('-----------')

                    if np.any(refine_flags):
                        refine_jdxs = group_jdxs[refine_flags]
                        refine_idxs, refine_conf = _entropy_refine(depth + 1, children, refine_jdxs)
                        # Overwrite course decisions with refined decisions.
                        refine_groupx = groupx[refine_flags]
                        pred_idxs[refine_groupx] = refine_idxs
                        pred_conf[refine_groupx] = refine_conf
            return pred_idxs, pred_conf

        nodes = sources
        jdxs = flat_jdxs
        depth = 0
        pred_idxs, pred_conf = _entropy_refine(depth, nodes, jdxs)
        return pred_idxs, pred_conf

    #### DEPRECATED METHODS

    def heirarchical_softmax(self, *args, **kw):
        import warnings
        warnings.warn('deprecated use the correctly spelled version')
        return self.hierarchical_softmax(*args, **kw)

    def heirarchical_cross_entropy(self, *args, **kw):
        import warnings
        warnings.warn('deprecated use the correctly spelled version')
        return self.hierarchical_cross_entropy(*args, **kw)

    def heirarchical_nll_loss(self, *args, **kw):
        import warnings
        warnings.warn('deprecated use the correctly spelled version')
        return self.hierarchical_nll_loss(*args, **kw)

    def heirarchical_log_softmax(self, *args, **kw):
        import warnings
        warnings.warn('deprecated use the correctly spelled version')
        return self.hierarchical_log_softmax(*args, **kw)


class CategoryTree(KWCOCO_CategoryTree, Mixin_CategoryTree_Torch):
    # Mixin the kwcoco category tree with torch functionality
    KWCOCO_CategoryTree.__doc__


def source_nodes(graph):
    """ generates source nodes --- nodes without incoming edges """
    return (n for n in graph.nodes() if graph.in_degree(n) == 0)


def sink_nodes(graph):
    """ generates source nodes --- nodes without incoming edges """
    return (n for n in graph.nodes() if graph.out_degree(n) == 0)


def gini(probs, axis=1, impl=np):
    """
    Approximates Shannon Entropy, but faster to compute

    Example:
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> import torch
        >>> rng = kwarray.ensure_rng(0)
        >>> probs = torch.softmax(torch.Tensor(rng.rand(3, 10)), 1)
        >>> gini(probs.numpy(), impl=kwarray.ArrayAPI.coerce('numpy'))
        array...0.896..., 0.890..., 0.892...
        >>> gini(probs, impl=kwarray.ArrayAPI.coerce('torch'))
        tensor...0.896..., 0.890..., 0.892...
    """
    return 1 - impl.sum(probs ** 2, axis=axis)


def entropy(probs, axis=1, impl=np):
    """
    Standard Shannon (Information Theory) Entropy

    Example:
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> import torch
        >>> rng = kwarray.ensure_rng(0)
        >>> probs = torch.softmax(torch.Tensor(rng.rand(3, 10)), 1)
        >>> entropy(probs.numpy(), impl=kwarray.ArrayAPI.coerce('numpy'))
        array...3.295..., 3.251..., 3.265...
        >>> entropy(probs, impl=kwarray.ArrayAPI.coerce('torch'))
        tensor...3.295..., 3.251..., 3.265...
    """
    with np.errstate(divide='ignore'):
        logprobs = impl.log2(probs)
        logprobs = impl.nan_to_num(logprobs, copy=False)
        h = -impl.sum(probs * logprobs, axis=axis)
        return h


