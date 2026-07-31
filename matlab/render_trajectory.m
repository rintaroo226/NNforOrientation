% render_trajectory.m
% generate_trajectory.py が出力した連続軌道 (t,qw,qx,qy,qz,wx,wy,wz の CSV) を
% 時系列順に読み込み、initBoxSim / renderBoxImage (無改変) でレンダリングして
% EKF のオフライン検証用データセットを作る。
%
% generate_random_database.m / generate_random_test_database.m はクォータニオン
% を内部で i.i.d. にサンプリングするが、このスクリプトは外部 (Python) で計算済み
% の連続軌道を読み込むだけで、initBoxSim/renderBoxImage には一切手を加えない。

addpath(fileparts(mfilename('fullpath')));  % initBoxSim, renderBoxImage

%% 設定
trajectory_csv = 'trajectory.csv';   % generate_trajectory.py の --out-csv と合わせる

cam_params.imageSize = 512;
cam_params.fov       = 25;
target_size          = [3, 2, 1];  % [幅, 高さ, 奥行] (m) generate_random_database.m と同一
ref_distance         = 10;

save_dir  = 'database_trajectory';
image_dir = fullfile(save_dir, 'images');

%% 出力先作成
[~, ~] = mkdir(save_dir);
[~, ~] = mkdir(image_dir);

%% 軌道 CSV を時系列順に読み込む
T = readtable(trajectory_csv);
N = height(T);

%% 初期化（1回だけ）
handles       = initBoxSim(cam_params.imageSize, target_size);
pose.distance = ref_distance;
pose.tx       = 0;
pose.ty       = 0;

%% レンダリングループ
csv_path = fullfile(save_dir, 'labels.csv');
fid = fopen(csv_path, 'w');
fprintf(fid, 'image,qw,qx,qy,qz,t\n');

bin_threshold = 5;  % これより明るい画素を前景(255)とみなす（train/test と同一）

fprintf('=== 軌道データベース生成 (%d枚) ===\n', N);
tic;

for i = 1:N
    q = [T.qw(i), T.qx(i), T.qy(i), T.qz(i)];
    [pose.pitch, pose.yaw, pose.roll] = quatToEulerZYX(q);

    img = renderBoxImage(handles, pose, cam_params);
    % 背景(黒)と前景(箱)のみの2値化シルエットとして保存
    bin_img = uint8((img > bin_threshold) * 255);

    rel_path = sprintf('images/frame_%06d.png', i);
    imwrite(bin_img, fullfile(save_dir, rel_path));

    fprintf(fid, '%s,%.8f,%.8f,%.8f,%.8f,%.6f\n', rel_path, q(1), q(2), q(3), q(4), T.t(i));

    if mod(i, 500) == 0
        fprintf('  %d/%d 完了 (%.1f秒)\n', i, N, toc);
    end
end

fclose(fid);
close(handles.fig);

fprintf('\n完了。%d枚 + ラベルを "%s/" に保存。(%.1f秒)\n', N, save_dir, toc);


function [pitch_deg, yaw_deg, roll_deg] = quatToEulerZYX(q)
% renderBoxImage.m の R = Rz(roll) * Ry(yaw) * Rx(pitch) 規約に合わせて
% クォータニオン [qw qx qy qz] を pitch/yaw/roll [deg] に変換する。
% generate_random_database.m の同名関数と同一 (rigid_body.py 側の
% quat_to_matrix と数値的に一致することはこのプロジェクトの検証で確認済み)。
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
