% render_trajectory_with_distance.m
% generate_trajectory_with_distance.py が出力した連続軌道
% (t,qw,qx,qy,qz,wx,wy,wz,distance の CSV) を時系列順に読み込み、
% initBoxSim/renderBoxImage (無改変) でレンダリングする。
%
% render_trajectory.m との唯一の違いは pose.distance をループ内で
% 毎フレーム CSV の distance 列から設定する点 (render_trajectory.m は
% ref_distance を全フレーム共通の定数として使う)。initBoxSim/renderBoxImage
% には一切手を加えない。
%
% labels.csv には距離が時間変化したことを後段の評価スクリプト
% (例: run_ekf_eval_bboxcrop.py) が誤差プロットの第2軸に使えるよう、
% generate_distance_sweep_database.m と同じ列名 "distance" も含める。

addpath(fileparts(mfilename('fullpath')));  % initBoxSim, renderBoxImage

%% 設定
trajectory_csv = 'trajectory_approach.csv';  % generate_trajectory_with_distance.py の --out-csv と合わせる

cam_params.imageSize = 512;
cam_params.fov       = 25;
target_size          = [3, 2, 1];  % [幅, 高さ, 奥行] (m) generate_random_database.m と同一

save_dir  = 'database_trajectory_with_distance';
image_dir = fullfile(save_dir, 'images');

%% 出力先作成
[~, ~] = mkdir(save_dir);
[~, ~] = mkdir(image_dir);

%% 軌道 CSV を時系列順に読み込む (distance 列を含む)
T = readtable(trajectory_csv);
N = height(T);
if ~ismember('distance', T.Properties.VariableNames)
    error('render_trajectory_with_distance:missing_distance', ...
        '%s に distance 列がありません。generate_trajectory_with_distance.py で生成したCSVを渡してください。', ...
        trajectory_csv);
end

%% 初期化（1回だけ）
handles = initBoxSim(cam_params.imageSize, target_size);
pose.tx = 0;
pose.ty = 0;

%% レンダリングループ (render_trajectory.m と異なり pose.distance を毎フレーム更新)
csv_path = fullfile(save_dir, 'labels.csv');
fid = fopen(csv_path, 'w');
fprintf(fid, 'image,qw,qx,qy,qz,t,distance\n');

bin_threshold = 5;  % render_trajectory.m と同一

fprintf('=== 距離変化軌道データベース生成 (%d枚) ===\n', N);
tic;

for i = 1:N
    q = [T.qw(i), T.qx(i), T.qy(i), T.qz(i)];
    [pose.pitch, pose.yaw, pose.roll] = quatToEulerZYX(q);
    pose.distance = T.distance(i);  % render_trajectory.m との唯一の差分

    img = renderBoxImage(handles, pose, cam_params);
    % 背景(黒)と前景(箱)のみの2値化シルエットとして保存
    bin_img = uint8((img > bin_threshold) * 255);

    rel_path = sprintf('images/frame_%06d.png', i);
    imwrite(bin_img, fullfile(save_dir, rel_path));

    fprintf(fid, '%s,%.8f,%.8f,%.8f,%.8f,%.6f,%.4f\n', ...
        rel_path, q(1), q(2), q(3), q(4), T.t(i), T.distance(i));

    if mod(i, 500) == 0
        fprintf('  %d/%d 完了 (%.1f秒, distance=%.1f)\n', i, N, toc, T.distance(i));
    end
end

fclose(fid);
close(handles.fig);

fprintf('\n完了。%d枚 + ラベルを "%s/" に保存。(%.1f秒)\n', N, save_dir, toc);


function [pitch_deg, yaw_deg, roll_deg] = quatToEulerZYX(q)
% render_trajectory.m の同名関数と同一 (renderBoxImage.m の
% R = Rz(roll) * Ry(yaw) * Rx(pitch) 規約に合わせた変換)。
w = q(1); x = q(2); y = q(3); z = q(4);

r11 = 1 - 2 * (y^2 + z^2);
r21 = 2 * (x * y + w * z);
r31 = 2 * (x * z - w * y);
r32 = 2 * (y * z + w * x);
r33 = 1 - 2 * (x^2 + y^2);

yaw_deg   = atan2d(-r31, sqrt(r11^2 + r21^2));
pitch_deg = atan2d(r32, r33);
roll_deg  = atan2d(r21, r11);
end
