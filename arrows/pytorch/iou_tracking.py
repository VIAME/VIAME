

class IOU_tracker(object):
    def __init__(self, iou_accept_threshold, iou_reject_threshold):
        self._iou_accept_threshold = iou_accept_threshold
        self._iou_reject_threshold = iou_reject_threshold
    
    def __call__(self, track_set, track_state_list):
        return self._track_iou(track_set, track_state_list)

    def _track_iou(self, track_set, track_state_list):
        # IOU based tracking
        for t_idx in range(len(track_set)):
            if len(track_state_list) > 0 and track_set[t_idx].active_flag is True:
                # get det with highest iou
                best_match = max(track_state_list, key=lambda x: self._iou_score(track_set[t_idx][-1].bbox, x.bbox))

                # sort the track state list in order to find the case that multiple bbox overlap with the current track
                sorted_ts_list = sorted(track_state_list, key=lambda x: self._iou_score(track_set[t_idx][-1].bbox, x.bbox))
                best_iou_score = self._iou_score(track_set[t_idx][-1].bbox, best_match.bbox)

                if best_iou_score >= self._iou_accept_threshold:

                    # if no other bbox with overlap larger than iou_reject_threshold
                    if len(sorted_ts_list) >= 2 and self._iou_score(track_set[t_idx][-1].bbox, sorted_ts_list[1].bbox) < self._iou_reject_threshold:
                        track_set[t_idx].updated_flag = True
                        track_set.update_track(track_set[t_idx].id, best_match)
    
                        # remove from best matching detection from detections
                        del track_state_list[track_state_list.index(best_match)]

        return track_set, track_state_list

    def _iou_score(self, bbox1, bbox2):
        """
        Calculates the intersection-over-union of two bounding boxes.
        
        Args:
            bbox1 list of float: bounding box in format x1,y1,w,h
            bbox2 list of float: bounding box in format x1,y1,w,h.
        
        Returns:
            int: intersection-over-onion of bbox1, bbox2
        """
        
        (x1, y1, w1, h1) = bbox1
        (x2, y2, w2, h2) = bbox2

        # get the overlap rectangle
        overlap_x0 = max(x1, x2)
        overlap_y0 = max(y1, y2)
        overlap_x1 = min(x1 + w1, x2 + w2)
        overlap_y1 = min(y1 + h1, y2 + h2)
        
        # check if there is an overlap
        if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
            return 0
        
        # if yes, calculate the ratio of the overlap to each ROI size and the unified size
        size_1 = w1 * h1
        size_2 = w2 * h2
        size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
        size_union = size_1 + size_2 - size_intersection
        
        return size_intersection / size_union
