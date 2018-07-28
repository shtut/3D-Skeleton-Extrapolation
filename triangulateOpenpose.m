function  [worldPoints, order] = triangulateOpenpose(calibration, fig)
    m = load(calibration);
    left_right_points = load(fig);
    worldPoints = triangulate(left_right_points.l,left_right_points.r,m.stereoParams);
    order = left_right_points.order;