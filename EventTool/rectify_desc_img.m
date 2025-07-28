clear all
close all

foldads = 'I:/DSEC';
dataname = 'zurich_city_11_c';

mkdir([foldads '/' dataname '/images_rectify'])
%% load image
img_ts_ads = sprintf([foldads '/' dataname '/timestamps.txt']);
image_ts = load(img_ts_ads);
frist_img_id = 1001; 
% last_img_id = length(image_ts)-1;
last_img_id = 1956;

frame = 1;
for i = frist_img_id:last_img_id
    image_ads = sprintf([foldads '/' dataname '/images/%06d.png'],i);% image id starts from 0
    img_raw(:,:,:,frame) = imread(image_ads);
    frame = frame + 1;
end

%% extrinsics - frame to event camera
% [T_10,R_rect0,R_rect1,K_rect0,K_rect1] = dsec_calibration_para(dataname)
calib_json = sprintf([foldads '/' dataname '/cam_to_cam.json']);
str = fileread(calib_json);
calib_para = jsondecode(str);
T_10 = calib_para.extrinsics.T_10;
R_rect0 = calib_para.extrinsics.R_rect0;
R_rect0 = [R_rect0 zeros(3,1)]; 
R_rect0 = [R_rect0;zeros(1,3) 1]; 
R_rect1 = calib_para.extrinsics.R_rect1;
R_rect1 = [R_rect1 zeros(3,1)]; 
R_rect1 = [R_rect1;zeros(1,3) 1]; 
k = calib_para.intrinsics.camRect0.camera_matrix;  
K_rect0 = [k(1) 0    k(3);
           0    k(2) k(4);
           0    0    1];
k = calib_para.intrinsics.camRect1.camera_matrix;  
K_rect1 = [k(1) 0    k(3);
           0    k(2) k(4);
           0    0    1];
      
T_rect1 = R_rect1*T_10;
reproject_position_x = zeros(1080,1440);
reproject_position_y = zeros(1080,1440);
reproj_img = zeros(480,640,size(img_raw,3),size(img_raw,4));
T = R_rect0*inv(T_rect1);

for x = 1:1440
    for y = 1:1080
        p1 = [x,y,1]';
        p1_u = inv(K_rect1)*p1;
        P1_n = p1_u/p1_u(3); % norm
        P1_n = [1e5 * P1_n;1];  % far
        P1_0 = T * P1_n;  % transfter p1 to 3D points in frame 0
        P1_0 = P1_0 / P1_0(end); % norm

        P1_0 = P1_0(1:3);
        p1_0 = K_rect0 * P1_0; 
        p1_0 = p1_0 / p1_0(end); % norm
        x_new = floor(p1_0(1)); 
        y_new = floor(p1_0(2));
        reproject_position_x(y,x) = x_new; % height
        reproject_position_y(y,x) = y_new; % width
        if x_new>0 && x_new<640 && y_new && y_new>0 && y_new<480
            reproj_img(y_new,x_new,:,:) = img_raw(y,x,:,:);
        end
    end         
end         

img_color = zeros(size(reproj_img,1),size(reproj_img,2),size(reproj_img,3),size(reproj_img,4));


%% rectify map
rectify_map_h5_ads = sprintf([foldads '/' dataname '/events/rectify_map.h5']);
rectify_map = double(h5read(rectify_map_h5_ads,'/rectify_map'));
map_x = reshape(rectify_map(1,:,:),[640,480]);
map_y = reshape(rectify_map(2,:,:),[640,480]);


for x = 1:640
    for y = 1:480
        y_new = round(map_y(x,y)); %[0,480]
        x_new = round(map_x(x,y)); %[0,640]
        if x_new > 0 && x_new < 640 && y_new > 0 && y_new < 480 
            img_color(y,x,:,:) = reproj_img(y_new,x_new,:,:);
        end
    end
end

frame = 1;
for i = frist_img_id:last_img_id
    image_ads = sprintf([foldads '/' dataname '/images_rectify/%06d.png'],i);% image id starts from 0
    imwrite(img_color(:,:,:,frame)/255,image_ads);
    frame = frame + 1;
end