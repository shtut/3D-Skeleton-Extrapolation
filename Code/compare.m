% Alex Nulman, Dvir Segal and Hadas Shahar [30-Apr-18]
%this function compares an openpose skeleton to a kinect one

% calibration - name of the calibration file containing matrices for the 2 cameras (stereo)
% fig1 - path to an openpose skeleton file as generated by skeletonize.py
% fig2 - path to a kinect skeleton as generated by skeletonize.py

function  [] = compare(calibration, fig1, fig2)
% load kinect data
kinect = load(fig2);
%calculate openpose 3D data
[OpenPoseSkeleton3D, orderOP] = triangulateOpenpose(calibration, fig1);
center = mean(OpenPoseSkeleton3D);
OpenPoseSkeleton3D = OpenPoseSkeleton3D - center;

% Split points to start and end points
OP_start = OpenPoseSkeleton3D(1:2:end,:);
OP_end = OpenPoseSkeleton3D(2:2:end,:);

kinect_start = kinect.lines(1:2:end,:);
kinect_end = kinect.lines(2:2:end,:);

orderK = cellstr(kinect.order);
orderOP = cellstr(orderOP);
% % % get kinect and openpose left foot point
OPhead = OpenPoseSkeleton3D(find(contains(orderOP,"Nose"),1,'first'),:);
OPneck = OpenPoseSkeleton3D(find(contains(orderOP,"Neck"),1,'first'),:);
OPshoulder1 = OpenPoseSkeleton3D(find(contains(orderOP,"LShoulder"),1,'first'),:);
OPshoulder2 = OpenPoseSkeleton3D(find(contains(orderOP,"RShoulder"),1,'first'),:);
OPelbow1 = OpenPoseSkeleton3D(find(contains(orderOP,"LElbow"),1,'first'),:);
OPelbow2 = OpenPoseSkeleton3D(find(contains(orderOP,"RElbow"),1,'first'),:);
OPhand1 = OpenPoseSkeleton3D(find(contains(orderOP,"LWrist"),1,'first'),:);
OPhand2 = OpenPoseSkeleton3D(find(contains(orderOP,"RWrist"),1,'first'),:);
OPhip1 = OpenPoseSkeleton3D(find(contains(orderOP,"LHip"),1,'first'),:);
OPhip2 = OpenPoseSkeleton3D(find(contains(orderOP,"RHip"),1,'first'),:);
OPknee1 = OpenPoseSkeleton3D(find(contains(orderOP,"LKnee"),1,'first'),:);
OPknee2 = OpenPoseSkeleton3D(find(contains(orderOP,"RKnee"),1,'first'),:);
OPfoot1 = OpenPoseSkeleton3D(find(contains(orderOP,"LAnkle"),1,'first'),:);
OPfoot2 = OpenPoseSkeleton3D(find(contains(orderOP,"RAnkle"),1,'first'),:);

Khead = kinect.lines(find(contains(orderK,"Neck"),1,'first'),:);
Kneck = kinect.lines(find(contains(orderK,"SpineShoulder"),1,'first'),:);
Kshoulder1 = kinect.lines(find(contains(orderK,"ShoulderLeft"),1,'first'),:);
Kshoulder2 = kinect.lines(find(contains(orderK,"ShoulderRight"),1,'first'),:);
Kelbow1 = kinect.lines(find(contains(orderK,"ElbowLeft"),1,'first'),:);
Kelbow2 =kinect.lines(find(contains(orderK,"ElbowRight"),1,'first'),:);
Khand1 = kinect.lines(find(contains(orderK,"WristLeft"),1,'first'),:);
Khand2 = kinect.lines(find(contains(orderK,"WristRight"),1,'first'),:);
Khip1 = kinect.lines(find(contains(orderK,"HipLeft"),1,'first'),:);
Khip2 = kinect.lines(find(contains(orderK,"HipRight"),1,'first'),:);
Kknee1 = kinect.lines(find(contains(orderK,"KneeLeft"),1,'first'),:);
Kknee2 = kinect.lines(find(contains(orderK,"KneeRight"),1,'first'),:);
Kfoot1 = kinect.lines(find(contains(orderK,"AnkleLeft"),1,'first'),:);
Kfoot2 = kinect.lines(find(contains(orderK,"AnkleRight"),1,'first'),:);

openposematch = [OPhead ;OPneck ;OPshoulder1 ;OPshoulder2 ;OPelbow1 ;OPelbow2 ;OPhand1 ;OPhand2 ;OPhip1 ;OPhip2 ;OPknee1 ;OPknee2 ;OPfoot1 ;OPfoot2];
kinectmatch = [Khead ;Kneck ;Kshoulder1 ;Kshoulder2 ;Kelbow1 ;Kelbow2 ;Khand1 ;Khand2 ;Khip1 ;Khip2 ;Kknee1 ;Kknee2 ;Kfoot1 ;Kfoot2];

% remove center of mass from all the points (move to 0,0)
center = mean(openposematch);
openposematch = openposematch - center;
openposematch = pointCloud(openposematch);

center = mean(kinectmatch);
kinectmatch = kinectmatch - center;
kinectmatch = pointCloud(kinectmatch);


% OP_start2 = openposematch.Location(1:2:end,:);
% OP_end2 = openposematch.Location(2:2:end,:);
% 
% kinect_start = kinectmatch.Location(1:2:end,:);
% kinect_end = kinectmatch.Location(2:2:end,:);

%calc transformation via icp
tform = pcregrigid(openposematch,kinectmatch,'Extrapolate',true);
%apply the transformation
openpose_transformed = pctransform(pointCloud(OpenPoseSkeleton3D),tform);
openpose_transformed = openpose_transformed.Location;

OP_start2 = openpose_transformed(1:2:end,:);
OP_end2 = openpose_transformed(2:2:end,:);



% OPfoot = OpenPoseSkeleton3D(find(contains(orderOP,"LAnkle"),1,'first'),:);
% % find the rotation matrix
% r = vrrotvec(Kfoot,OPfoot);
% m = vrrotvec2mat(r);
% OP_end2 = OP_end * m; OP_start2 = OP_start * m;
% % 
% % find the scaling factor
% OPfoot_postR = OPfoot * m;
% Kfoot_postR = Kfoot * m;
% scale = (norm(Kfoot_postR - [0,0,0]))/(norm(OPfoot_postR - [0,0,0]));
% OP_end2 = OP_end2 * scale; OP_start2 = OP_start2 * scale;

% Concat each start and end point to [[start, end] ,...[start, end]] list
%  then  plot 3D points on viewer with 
% 3D line between each point
figure('Name', 'fig1')
for elm = transpose(cat(2, OP_start2,OP_end2))
    plot3([elm(1);elm(4)], [elm(2);elm(5)], [elm(3);elm(6)],  'Color', [255, 153, 51] / 255, 'MarkerSize', 30, 'LineWidth', 3);
    scatter3([elm(1);elm(4)], [elm(2);elm(5)], [elm(3);elm(6)],'MarkerFaceColor',[255, 153, 51] / 255, 'MarkerEdgeColor','none');
	hold on;
	grid on;
    % height limited to +/-3m, width to +/-2m, depth to +/-1m
    axis([-5 5 -5 5 -5 5]);
end
for elm = transpose(cat(2, kinect_start,kinect_end))
    plot3([elm(1);elm(4)], [elm(2);elm(5)], [elm(3);elm(6)],  'Color', [51, 153, 255] / 255, 'MarkerSize', 30, 'LineWidth', 3);
    scatter3([elm(1);elm(4)], [elm(2);elm(5)], [elm(3);elm(6)],'MarkerFaceColor',[51, 153, 255] / 255, 'MarkerEdgeColor','none');
	hold on; grid on;
    axis([-5 5 -5 5 -5 5]);
end
