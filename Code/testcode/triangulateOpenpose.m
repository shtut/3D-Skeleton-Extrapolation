% Alex Nulman, Dvir Segal and Hadas Shahar [30-Apr-18]

% given calibration and a figure containing 2 OpenPose skeletons (2D) - generates a 3D OpenPose skeleton
% calibration - name of the calibration file containing matrices for the 2 cameras (stereo)
% fig - name of the text file containing both skeletons
function  [worldPoints, order] = triangulateOpenpose(calibration, fig)
    m = load(calibration);
    left_right_points = load(fig);
    left = left_right_points.l;
    right = left_right_points.r;
    
    left = [1033, 287; 1021, 374; 1021, 374; 934, 352; 1021, 374; 1051, 379; 934, 352; 822, 369; 822, 369; 858, 484; 1051, 379; 1119, 435; 1119, 435; 1131, 340; 1021, 374; 990, 575; 1021, 374; 1076, 576; 990, 575; 967, 790; 967, 790; 956, 992; 1076, 576; 1049, 792; 1049, 792; 1037, 990];
    right = [936, 336; 933, 411; 933, 411; 865, 409; 933, 411; 992, 409; 865, 409; 777, 448; 777, 448; 808, 537; 992, 409; 1097, 438; 1097, 438; 1093, 347; 933, 411; 888, 581; 933, 411; 979, 579; 888, 581; 896, 773; 896, 773; 899, 950; 979, 579; 976, 773; 976, 773; 980, 955];

    
    left(:,1) = left(:,1) /1000;
    left(:,2) = left(:,2)  /1000;
    right(:,1) = right(:,1)  /1000;
    right(:,2) = right(:,2)  /1000;
    
    worldPoints = triangulate(left,right,m.stereoParams);
    order = left_right_points.order; 
    order = ['head','neck','neck','shoulder1','neck','shoulder2','shoulder1','elbow1','elbow1','wrist1','shoulder2','elbow2','elbow2','wrist2','neck','hip1','neck','hip2','hip1','knee1','knee1','foot1','hip2','knee2','knee2','foot2'];