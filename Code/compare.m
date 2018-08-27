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
OpenPoseSkeleton3D = OpenPoseSkeleton3D - OpenPoseSkeleton3D(1:1,:);

% Split points to start and end points
OP_start2 = OpenPoseSkeleton3D(1:2:end,:);
OP_end2 = OpenPoseSkeleton3D(2:2:end,:);

kinect_start = kinect.lines(1:2:end,:);
kinect_end = kinect.lines(2:2:end,:);

% orderK = cellstr(kinect.order);
% orderOP = cellstr(orderOP);
% % % get kinect and openpose left foot point
% Kfoot = kinect.lines(find(contains(orderK,"FootLeft"),1,'first'),:);
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
