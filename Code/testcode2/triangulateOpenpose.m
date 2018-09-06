% Alex Nulman, Dvir Segal and Hadas Shahar [30-Apr-18]

% given calibration and a figure containing 2 OpenPose skeletons (2D) - generates a 3D OpenPose skeleton
% calibration - name of the calibration file containing matrices for the 2 cameras (stereo)
% fig - name of the text file containing both skeletons
function  [worldPoints, order] = triangulateOpenpose(calibration, fig)
    m = load(calibration);
    left_right_points = load(fig);
    left = left_right_points.l;
    right = left_right_points.r;
            
    worldPoints = triangulate(left,right,m.stereoParams);
    order = left_right_points.order; 
