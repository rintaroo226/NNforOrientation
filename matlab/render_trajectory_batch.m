% render_trajectory_batch.m
% generate_trajectory_batch.py が出力した複数本の軌道 CSV (traj_000.csv, ...)
% をまとめてレンダリングする。render_trajectory.m の複数本版で、
% initBoxSim/renderBoxImage (無改変) を全軌道で使い回す。

addpath(fileparts(mfilename('fullpath')));  % initBoxSim, renderBoxImage

%% 設定
% generate_trajectory_batch.py --out-dir trajectories で生成した
% trajectories/ フォルダを、この .m ファイルと同じ matlab/ 直下にコピーしてから実行する
% (render_trajectory.m を trajectory.csv をコピーして使っていたのと同じ運用)。
trajectory_dir = 'trajectories';
save_root      = 'database_trajectory_batch';

cam_params.imageSize = 512;
cam_params.fov       = 25;
target_size          = [3, 2, 1];  % [幅, 高さ, 奥行] (m) generate_random_database.m と同一
ref_distance         = 10;

bin_threshold = 5;  % これより明るい画素を前景(255)とみなす（train/test と同一）

%% 軌道 CSV 一覧を取得
csv_files = dir(fullfile(trajectory_dir, 'traj_*.csv'));
csv_files = csv_files(~[csv_files.isdir]);
[~, order] = sort({csv_files.name});
csv_files = csv_files(order);

fprintf('=== 軌道バッチレンダリング (%d本) ===\n', numel(csv_files));

%% 初期化（全軌道で使い回す。1回だけ）
handles       = initBoxSim(cam_params.imageSize, target_size);
pose.distance = ref_distance;
pose.tx       = 0;
pose.ty       = 0;

for ti = 1:numel(csv_files)
    [~, traj_name, ~] = fileparts(csv_files(ti).name);
    traj_csv = fullfile(csv_files(ti).folder, csv_files(ti).name);

    save_dir  = fullfile(save_root, traj_name);
    image_dir = fullfile(save_dir, 'images');
    [~, ~] = mkdir(save_dir);
    [~, ~] = mkdir(image_dir);

    T = readtable(traj_csv);
    N = height(T);

    csv_path = fullfile(save_dir, 'labels.csv');
    fid = fopen(csv_path, 'w');
    fprintf(fid, 'image,qw,qx,qy,qz,t\n');

    fprintf('[%d/%d] %s (%d枚) ', ti, numel(csv_files), traj_name, N);
    tic;

    for i = 1:N
        q = [T.qw(i), T.qx(i), T.qy(i), T.qz(i)];
        [pose.pitch, pose.yaw, pose.roll] = quatToEulerZYX(q);

        img = renderBoxImage(handles, pose, cam_params);
        bin_img = uint8((img > bin_threshold) * 255);

        rel_path = sprintf('images/frame_%06d.png', i);
        imwrite(bin_img, fullfile(save_dir, rel_path));

        fprintf(fid, '%s,%.8f,%.8f,%.8f,%.8f,%.6f\n', rel_path, q(1), q(2), q(3), q(4), T.t(i));
    end

    fclose(fid);
    fprintf('完了 (%.1f秒)\n', toc);
end

close(handles.fig);
fprintf('\n全%d本のレンダリングが完了しました。出力先: %s/\n', numel(csv_files), save_root);


function [pitch_deg, yaw_deg, roll_deg] = quatToEulerZYX(q)
% render_trajectory.m / generate_random_database.m と同一の変換。
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
