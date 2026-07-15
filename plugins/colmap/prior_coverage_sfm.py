#!/usr/bin/env python3
"""prior_coverage_sfm.py - COLMAP rig-SfM backend for detect_prior_coverage.

Experimental alternative geometry engine (--method sfm-rig): instead of 2D
registration chains, runs COLMAP incremental SfM with the three rig cameras
constrained as a fixed multi-camera rig (pycolmap >= 3.12 RigConfig; images
taken at the same trigger become one rig "frame"). Camera poses are then
converted to per-image ground-plane homographies:

  1. symlink tree with per-camera folders and trigger-normalized filenames
     (COLMAP groups rig frames by the image name after the camera prefix,
     and SSL filenames embed the camera letter, so SLC/SLP/SLS must be
     renamed to a shared per-trigger name);
  2. SIFT extraction + sequential matching + rig-constrained incremental
     mapping;
  3. robust plane fit to the sparse points (the survey scene is near-planar
     coastline) and per-image image->plane homographies by ray casting the
     image corners onto the plane;
  4. similarity fit plane->ENU from the per-frame camera centres vs GPS, so
     results land in the same shared ENU frame / coverage grid as the other
     methods. Images SfM could not register (featureless open water) fall
     back to the metadata footprint transform.

The coverage computation, outputs and visualizations are shared with
detect_prior_coverage.py.
"""

import math
import os
import shutil
import time

import numpy as np


def _plane_basis(points):
    """Robust plane fit. Returns (origin, ex, ey, normal) with two trims of
    the farthest outliers (sparse SfM clouds contain sky/water junk)."""
    pts = np.asarray(points)
    keep = np.ones(len(pts), dtype=bool)
    for _ in range(3):
        p = pts[keep]
        c = p.mean(axis=0)
        _u, _s, vt = np.linalg.svd(p - c, full_matrices=False)
        normal = vt[2]
        d = np.abs((pts - c) @ normal)
        thr = max(3.0 * np.median(d[keep]), 1e-6)
        keep = d <= thr
    ex, ey = vt[0], vt[1]
    return c, ex, ey, normal


def _image_to_plane_h(image, camera, origin, ex, ey, normal, w, h):
    """Homography mapping image pixels -> 2D plane coordinates by casting
    the corner rays onto the plane. Returns None for degenerate geometry
    (plane behind camera / grazing rays)."""
    import cv2
    cam_from_world = image.cam_from_world()
    R = cam_from_world.rotation.matrix()
    t = cam_from_world.translation
    C = -R.T @ t                       # camera centre in world
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                       dtype=np.float64)
    plane_pts = []
    for px in corners:
        ray_cam = camera.cam_from_img(px)
        d = R.T @ np.array([ray_cam[0], ray_cam[1], 1.0])
        denom = d @ normal
        if abs(denom) < 1e-9:
            return None
        s = ((origin - C) @ normal) / denom
        if s <= 0:
            return None
        X = C + s * d
        plane_pts.append([(X - origin) @ ex, (X - origin) @ ey])
    return cv2.getPerspectiveTransform(
        corners.astype(np.float32), np.array(plane_pts, dtype=np.float32))


def _fit_similarity_2d(src, dst):
    """Least-squares similarity (rot+scale+trans, both chiralities)
    src->dst for (N,2) arrays. Returns 3x3."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    s0, d0 = src - cs, dst - cd
    best = None
    for chir in (1, -1):
        s_ref = s0.copy()
        s_ref[:, 1] *= chir
        num = (d0[:, 0] * s_ref[:, 0] + d0[:, 1] * s_ref[:, 1]).sum(), \
              (d0[:, 1] * s_ref[:, 0] - d0[:, 0] * s_ref[:, 1]).sum()
        den = (s_ref ** 2).sum()
        a, b = num[0] / den, num[1] / den
        M = np.array([[a, -b], [b, a]]) @ np.diag([1.0, chir])
        dst_hat = s0 @ M.T + cd    # full map: dst_hat = M @ (src - cs) + cd
        res = float(np.median(np.linalg.norm(dst_hat - dst, axis=1)))
        if best is None or res < best[1]:
            best = (M, res)
    M, res = best
    T = np.eye(3)
    T[:2, :2] = M
    T[:2, 2] = cd - M @ cs
    return T, res


def run(args):
    """Entry point called by detect_prior_coverage.main for --method sfm-rig.

    The coverage grid and output writers still live in the detect_prior_coverage
    tool (on sys.path via the calling tool); the registration core and metadata
    reader now live in the viame.opencv plugin. All are imported lazily so this
    module stays importable standalone.
    """
    import pycolmap
    import detect_prior_coverage as dpc
    from viame.opencv import survey_metadata as smd

    grid = dpc.CoverageGrid(cell_m=args.grid_cell)
    origin_ref = {'lat': None, 'lon': None, 'to_enu': None}
    obs_registry = {}
    order = 0

    for site_id, site_folder in enumerate(args.sites):
        t0 = time.time()
        site_tag = os.path.basename(os.path.normpath(site_folder))
        print(f'\n=== {site_tag} (sfm-rig) ===')
        records, cams = smd.build_image_records(
            site_folder, flight_logs=args.flight_logs, read_exif=True)
        day = smd.folder_date(site_folder) or ''

        out_dir = args.output or (os.path.normpath(site_folder) + '_coverage')
        if args.output and len(args.sites) > 1:
            out_dir = os.path.join(args.output, site_tag)
        os.makedirs(out_dir, exist_ok=True)
        work = os.path.join(out_dir, 'sfm_work')
        img_root = os.path.join(work, 'images')
        if os.path.isdir(img_root):
            shutil.rmtree(img_root)

        # --- image tree with trigger-normalized names. Hardlinks (fallback:
        # copies) rather than symlinks: COLMAP stores symlink targets as the
        # image name, which breaks the rig image_prefix grouping. ---
        name_map = {}       # colmap image name -> (cam, rel)
        for cam, rels in cams.items():
            cam_dir = os.path.join(img_root, cam or 'MONO')
            os.makedirs(cam_dir, exist_ok=True)
            for rel in rels:
                info = smd.parse_image_filename(rel)
                fr = info['frame']
                if fr is None:
                    continue
                nm = f'frame_{fr:06d}.jpg'
                src = os.path.realpath(os.path.join(site_folder, rel))
                dst = os.path.join(cam_dir, nm)
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
                name_map[f'{cam or "MONO"}/{nm}'] = (cam, rel)

        db_path = os.path.join(work, 'database.db')
        if os.path.exists(db_path):
            os.remove(db_path)
        extraction = pycolmap.FeatureExtractionOptions()
        extraction.max_image_size = 3200
        print('  Extracting features...')
        pycolmap.extract_features(
            db_path, img_root, camera_mode=pycolmap.CameraMode.PER_FOLDER,
            extraction_options=extraction)
        # --- GPS pose priors: the rig cameras share the aircraft position
        # (their projection centres are metres apart), so every image of a
        # trigger gets the same WGS84 prior. They serve twice: spatial
        # match pairing below, and with use_prior_position the bundle
        # adjustment is anchored to GPS, controlling drift and returning
        # the reconstruction in a metric local frame. ---
        db = pycolmap.Database.open(db_path)
        db.clear_pose_priors()
        n_priors = 0
        cov = np.diag([20.0 ** 2, 20.0 ** 2, 30.0 ** 2])
        for name, (cam, rel) in name_map.items():
            rmeta = records.get(rel, {})
            if rmeta.get('lat') is None:
                continue
            try:
                image = db.read_image_with_name(name)
            except Exception:
                continue
            prior = pycolmap.PosePrior()
            prior.corr_data_id = pycolmap.data_t(
                sensor_id=pycolmap.sensor_t(
                    type=pycolmap.SensorType.CAMERA, id=image.camera_id),
                id=image.image_id)
            prior.position = np.array([
                rmeta['lat'], rmeta['lon'], rmeta.get('alt_agl') or 0.0])
            prior.position_covariance = cov
            prior.coordinate_system = \
                pycolmap.PosePriorCoordinateSystem.WGS84
            db.write_pose_prior(prior)
            n_priors += 1
        db.close()
        if n_priors:
            print(f'  Wrote {n_priors} GPS pose priors')

        # Matching strategy. With GPS, sequential + spatial (GPS-proximity)
        # pairing already finds every loop-closure / crossing-pass pair that
        # exhaustive would, at O(n*k) instead of O(n^2) cost and without the
        # far-apart water pairs that risk false matches. Exhaustive is worth
        # it only WITHOUT GPS (no spatial pairing to catch revisits) and on
        # small sites.
        n_imgs = len(name_map)
        matching = getattr(args, 'sfm_matching', 'auto')
        use_exhaustive = (matching == 'exhaustive'
                          or (matching == 'auto' and n_priors < 4
                              and n_imgs <= 300))
        if use_exhaustive:
            print(f'  Matching (exhaustive, {n_imgs} images)...')
            pycolmap.match_exhaustive(db_path)
        else:
            print('  Matching (sequential)...')
            pairing = pycolmap.SequentialPairingOptions()
            pairing.overlap = 10
            pairing.quadratic_overlap = True
            pairing.expand_rig_images = True
            pycolmap.match_sequential(db_path, pairing_options=pairing)
            if n_priors >= 4 and matching != 'sequential':
                # Spatial matching pairs images by GPS proximity - the ONLY
                # matcher that connects revisits/crossing passes hundreds of
                # triggers apart (sequential pairing never sees them).
                print('  Matching (spatial, GPS-prior based)...')
                try:
                    spatial = pycolmap.SpatialPairingOptions()
                    spatial.max_distance = 150.0   # ~footprint radius (m)
                    spatial.max_num_neighbors = 24
                    spatial.ignore_z = True
                    pycolmap.match_spatial(db_path, pairing_options=spatial)
                except (RuntimeError, ValueError) as e:
                    print(f'    spatial matching unavailable ({e})')

        sparse_dir = os.path.join(work, 'sparse')
        os.makedirs(sparse_dir, exist_ok=True)

        def _pipeline_opts(refine_rig):
            opts = pycolmap.IncrementalPipelineOptions()
            opts.ba_refine_sensor_from_rig = refine_rig
            if n_priors >= 4:
                opts.use_prior_position = True
                opts.use_robust_loss_on_prior_position = True
            # High-altitude nadir surveys have tiny triangulation angles
            # (~29 m trigger baseline at ~250 m AGL -> ~6 deg); COLMAP's
            # terrestrial defaults would reject every valid init pair and
            # most absolute-pose registrations.
            opts.mapper.init_min_tri_angle = 4.0
            opts.mapper.abs_pose_min_num_inliers = 15
            return opts

        # --- COLMAP >= 4 rig workflow: the incremental mapper requires
        # known cam_from_rig extrinsics, so (1) bootstrap WITHOUT rig
        # constraints, (2) derive the rig extrinsics from that
        # reconstruction via apply_rig_config, (3) re-map rig-constrained -
        # whole frames then register as a unit, giving water images a pose
        # whenever a sibling camera registers. ---
        import sqlite3
        con = sqlite3.connect(db_path)
        for table in ('frame_data', 'frames', 'rig_sensors', 'rigs'):
            con.execute(f'DELETE FROM {table}')
        con.commit()
        con.close()
        print('  Bootstrap mapping (no rig constraints)...')
        recs = pycolmap.incremental_mapping(
            db_path, img_root, sparse_dir, options=_pipeline_opts(False))
        if not recs:
            print('  SfM produced no reconstruction; skipping site')
            continue
        rec = max(recs.values(), key=lambda r: r.num_reg_images())
        print(f'    bootstrap: {rec.num_reg_images()}/{len(name_map)} '
              f'images, {rec.num_points3D()} points')

        # Only cameras with a registered bootstrap image can have their
        # cam_from_rig derived; one unknown sensor invalidates the whole
        # rig, so the others are left as independent sensors.
        reg_cams = {name_map[im.name][0] for im in rec.images.values()
                    if im.has_pose and im.name in name_map}
        rig_members = [c for c in sorted(cams, key=lambda c: dpc.CAM_ORDER[c])
                       if c in reg_cams]
        if len(cams) > 1 and len(rig_members) > 1:
            if len(rig_members) < len(cams):
                print(f'    rig limited to {"/".join(rig_members)} '
                      f'(others unregistered in bootstrap)')
            ref = 'CENTER' if 'CENTER' in rig_members else rig_members[0]
            rig_cams = []
            for cam in rig_members:
                rig_cams.append(pycolmap.RigConfigCamera(
                    image_prefix=f'{cam or "MONO"}/',
                    ref_sensor=(cam == ref)))
            configs = [pycolmap.RigConfig(cameras=rig_cams)]
            # Every image must belong to a rig; cameras that could not
            # join (unregistered in bootstrap) become their own trivial
            # single-sensor rigs.
            for cam in cams:
                if cam not in rig_members:
                    configs.append(pycolmap.RigConfig(cameras=[
                        pycolmap.RigConfigCamera(
                            image_prefix=f'{cam or "MONO"}/',
                            ref_sensor=True)]))
            try:
                db = pycolmap.Database.open(db_path)
                pycolmap.apply_rig_config(configs, db, rec)
                db.close()
                print('  Rig-constrained mapping (extrinsics from '
                      'bootstrap)...')
                rig_sparse = os.path.join(work, 'sparse_rig')
                os.makedirs(rig_sparse, exist_ok=True)
                rig_recs = pycolmap.incremental_mapping(
                    db_path, img_root, rig_sparse,
                    options=_pipeline_opts(True))
                if rig_recs:
                    rig_rec = max(rig_recs.values(),
                                  key=lambda r: r.num_reg_images())
                    if rig_rec.num_reg_images() >= rec.num_reg_images():
                        rec = rig_rec
                    else:
                        print('    rig-constrained run registered fewer '
                              'images; keeping bootstrap result')
                else:
                    print('    rig-constrained run produced nothing; '
                          'keeping bootstrap result')
            except (RuntimeError, ValueError, IndexError) as e:
                print(f'    rig stage unavailable ({e}); keeping bootstrap')
        print(f'  Registered {rec.num_reg_images()}/{len(name_map)} images, '
              f'{rec.num_points3D()} points')
        # Report the refined rig extrinsics (metric when GPS priors were
        # used). The rig cameras are co-located, so the translation should
        # be ~0 m; the across-track cant lives in the rotation angle
        # (expect ~20 deg for PORT/STAR: ~90 m offset at ~250 m AGL).
        for rig in rec.rigs.values():
            for sid, sfr in rig.non_ref_sensors.items():
                if sfr is None:
                    continue
                ang = 2.0 * math.degrees(math.acos(
                    min(1.0, abs(float(sfr.rotation.quat[3])))))
                print(f'    rig camera {sid.id}: cant {ang:.1f} deg '
                      f'from reference')

        # Robust similarity alignment of the reconstruction to the GPS
        # track (in a local metric ENU frame). Prior-anchored BA alone has
        # been observed to leave large scale errors on weak planar scenes;
        # an explicit RANSAC alignment to the camera GPS positions fixes
        # scale/orientation deterministically.
        fixes = [(r, records[r]) for cam in cams for r in cams[cam]
                 if records.get(r, {}).get('lat') is not None]
        if fixes and origin_ref['lat'] is None:
            origin_ref['lat'] = fixes[0][1]['lat']
            origin_ref['lon'] = fixes[0][1]['lon']
            origin_ref['to_enu'] = smd.make_enu(origin_ref['lat'],
                                                origin_ref['lon'])
        _to_enu = origin_ref['to_enu']
        if _to_enu is not None:
            rel_to_name = {rel: nm for nm, (c, rel) in name_map.items()}
            tgt_names, tgt_locs = [], []
            for rel, rmeta in fixes:
                nm = rel_to_name.get(rel)
                if nm is None:
                    continue
                e, n = _to_enu(rmeta['lat'], rmeta['lon'])
                tgt_names.append(nm)
                tgt_locs.append([e, n, rmeta.get('alt_agl') or 0.0])
            try:
                ransac = pycolmap.RANSACOptions()
                ransac.max_error = 25.0     # metres, ~GPS + lever arm
                sim = pycolmap.align_reconstruction_to_locations(
                    rec, tgt_names, np.array(tgt_locs), 3, ransac)
                if sim is not None:
                    rec.transform(sim)
                    print(f'  Aligned reconstruction to GPS track '
                          f'(scale {sim.scale:.3f})')
            except (RuntimeError, ValueError) as e:
                print(f'    GPS alignment failed ({e})')

        pts = np.array([p.xyz for p in rec.points3D.values()])
        origin, ex, ey, normal = _plane_basis(pts)

        # --- per-image plane homographies + plane->ENU similarity ---
        plane_T = {}    # rel -> 3x3 image->plane
        cam_centers = []
        enu_pts = []
        first_fix = next((records[r] for cam in cams for r in cams[cam]
                          if records.get(r, {}).get('lat') is not None), None)
        if first_fix is not None and origin_ref['lat'] is None:
            origin_ref['lat'], origin_ref['lon'] = (first_fix['lat'],
                                                    first_fix['lon'])
            origin_ref['to_enu'] = smd.make_enu(origin_ref['lat'],
                                                origin_ref['lon'])
        to_enu = origin_ref['to_enu']
        for im in rec.images.values():
            key = im.name
            if key not in name_map or not im.has_pose:
                continue
            cam, rel = name_map[key]
            rec_meta = records.get(rel, {})
            w = rec_meta.get('width') or 5168
            h = rec_meta.get('height') or 3448
            camera = rec.cameras[im.camera_id]
            H = _image_to_plane_h(im, camera, origin, ex, ey, normal, w, h)
            if H is None:
                continue
            plane_T[rel] = H
            if rec_meta.get('lat') is not None and to_enu is not None:
                cfw = im.cam_from_world()
                C = -cfw.rotation.matrix().T @ cfw.translation
                cam_centers.append([(C - origin) @ ex, (C - origin) @ ey])
                enu_pts.append(to_enu(rec_meta['lat'], rec_meta['lon']))

        S = None
        if len(cam_centers) >= 3:
            S, res = _fit_similarity_2d(cam_centers, enu_pts)
            scale = float(np.sqrt(abs(np.linalg.det(S[:2, :2]))))
            print(f'  Plane->ENU similarity: scale {scale:.3f}, '
                  f'median residual {res:.1f} m over {len(cam_centers)} '
                  f'camera centres')
            # With GPS priors the reconstruction is already metric, so the
            # fitted scale must be ~1; a strong deviation on a small basis
            # means the fit (or the reconstruction) is untrustworthy -
            # prefer the metadata footprints for everything.
            if n_priors >= 4 and (len(cam_centers) < 4
                                  or abs(scale - 1.0) > 0.3):
                print('    similarity inconsistent with GPS-anchored '
                      'reconstruction; using metadata footprints instead')
                S = None
        elif to_enu is None:
            # No GPS anywhere: use plane coords directly as pseudo-ENU.
            S = np.eye(3)
            print('  No GPS: coverage grid uses SfM plane units')

        # --- observations ---
        frames_by_cam, idx_by_cam = {}, {}
        for cam, rels in cams.items():
            fr = [smd.parse_image_filename(r)['frame'] for r in rels]
            fr = [f if f is not None else i for i, f in enumerate(fr)]
            frames_by_cam[cam] = fr
            idx_by_cam[cam] = {f: i for i, f in enumerate(fr)}
        observations = []
        triggers = sorted({f for fr in frames_by_cam.values() for f in fr})
        n_sfm = n_meta = 0
        for t in triggers:
            for cam in sorted(cams, key=lambda c: dpc.CAM_ORDER[c]):
                i = idx_by_cam[cam].get(t)
                if i is None:
                    continue
                rel = cams[cam][i]
                rmeta = records.get(rel, {})
                w = rmeta.get('width') or 5168
                h = rmeta.get('height') or 3448
                T = None
                if S is not None and rel in plane_T:
                    T = S @ plane_T[rel]
                    n_sfm += 1
                elif rmeta.get('lat') is not None and to_enu is not None:
                    poses = {r: records[r] for r in cams[cam]
                             if records.get(r, {}).get('lat') is not None}
                    from viame.opencv.registration_utils import (
                        _poses_to_enu, _track_headings)
                    enu_local, _ = _poses_to_enu(poses, cams[cam])
                    heads = _track_headings(enu_local)
                    heading = heads[i]
                    lat_frac = {'PORT': -args.xcam_offset_frac,
                                'CENTER': 0.0,
                                'STAR': args.xcam_offset_frac}.get(cam, 0.0)
                    T = dpc._metadata_transform(
                        rmeta, 0.0 if np.isnan(heading) else heading,
                        w, h, to_enu, lat_frac)
                    if T is not None:
                        n_meta += 1
                observations.append(dpc.Observation(
                    order=order, site_id=site_id, site_tag=site_tag,
                    site_dir=site_folder, cam=cam,
                    frame=rmeta.get('frame', t), rel=rel, width=w, height=h,
                    T_enu=T, chain_H=None, timestep=t, is_water=False,
                    pass_no=rmeta.get('pass') or 1, day=day,
                    has_gps=rmeta.get('lat') is not None))
                order += 1
        for o in observations:
            obs_registry[o.order] = o
        print(f'  {len(observations)} images: {n_sfm} SfM-referenced, '
              f'{n_meta} metadata fallback')

        rows, revisit_events, frac = dpc.compute_coverage(
            observations, grid, args, {}, {}, frames_by_cam, idx_by_cam,
            site_folder, {}, obs_registry)

        dpc.write_viame_csv(os.path.join(out_dir, 'prior_coverage.csv'),
                            rows, args.coverage_class, obs_registry)
        dpc.write_revisits_csv(os.path.join(out_dir, 'revisits.csv'),
                               revisit_events)
        dpc.render_coverage_map(os.path.join(out_dir, 'coverage_map.png'),
                                observations, site_tag)
        if not args.no_thumbnails:
            dpc.render_thumbnail_grid(
                os.path.join(out_dir, 'prior_coverage_vis.png'),
                site_folder, observations, rows, {},
                obs_registry=obs_registry,
                max_rows=getattr(args, 'vis_rows', 40),
                thumb_w=getattr(args, 'vis_thumb_width', 420),
                frames=dpc.parse_frame_range(getattr(args, 'vis_frames', None)))
        by_cam = {}
        for o in observations:
            by_cam.setdefault(o.cam, []).append(frac.get(o.rel, 0.0))
        for cam in sorted(by_cam, key=lambda c: dpc.CAM_ORDER[c]):
            v = by_cam[cam]
            n_any = sum(1 for x in v if x > 0.01)
            print(f'    {cam or "MONO"}: {n_any}/{len(v)} frames with prior '
                  f'coverage, '
                  f'mean seen fraction {np.mean(v):.2f}')
        print(f'  Outputs -> {out_dir}  ({time.time() - t0:.0f}s)')
    print(f'\nDone: {order} images')
