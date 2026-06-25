function img = renderBoxImage(handles, pose, cam_params)
% 指定された姿勢でターゲットを描画し、グレースケール画像を返す
%
% handles   : initBoxSim の戻り値
% pose      : struct(pitch, yaw, roll [deg], distance, tx, ty [m])
% cam_params: struct(fov [deg], imageSize [px])

% 回転行列 (Ry * Rx * Rz の順: yaw → pitch → roll)
Rx = [1, 0,               0;
      0, cosd(pose.pitch), -sind(pose.pitch);
      0, sind(pose.pitch),  cosd(pose.pitch)];

Ry = [ cosd(pose.yaw), 0, sind(pose.yaw);
       0,              1, 0;
      -sind(pose.yaw), 0, cosd(pose.yaw)];

Rz = [cosd(pose.roll), -sind(pose.roll), 0;
      sind(pose.roll),  cosd(pose.roll), 0;
      0,                0,               1];

R = Rz * Ry * Rx;  % pitch → yaw → roll_camera（カメラ Z 軸周りの in-plane 回転）

% 頂点を回転・並進させて配置
pos = [pose.tx; pose.ty; pose.distance];
new_v = (R * handles.v0')' + pos';

% 各面の patch を更新
for k = 1:length(handles.body)
    handles.body(k).Vertices = new_v;
end

% カメラ設定
set(handles.ax, 'CameraPosition', [0 0 0], ...
                'CameraTarget',   [0 0 1], ...
                'CameraViewAngle', cam_params.fov);

% フレームキャプチャ
frame = getframe(handles.fig);
img   = frame2im(frame);

% グレースケール変換
if size(img, 3) == 3
    img = rgb2gray(img);
end

% 解像度を cam_params.imageSize に統一（Retina 対策）
target_sz = cam_params.imageSize;
if size(img, 1) ~= target_sz || size(img, 2) ~= target_sz
    img = imresize(img, [target_sz, target_sz]);
end

end
