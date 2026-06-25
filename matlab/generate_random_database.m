% generate_random_database.m
% pitch/yaw/roll をランダムサンプリングして学習用データベースを生成する

addpath(fileparts(mfilename('fullpath')));  % initBoxSim, renderBoxImage

%% 設定
N            = 10000;       % サンプル数
pitch_range  = [0, 180];    % pitch 範囲 [deg]
yaw_range    = [0, 360];    % yaw 範囲 [deg]
roll_range   = [0, 360];    % roll 範囲 [deg]
rng_seed     = 42;

cam_params.imageSize = 512;
cam_params.fov       = 25;
target_size          = [3, 2, 1];  % [幅, 高さ, 奥行] (m)
ref_distance         = 10;

save_dir  = 'database_random';
image_dir = fullfile(save_dir, 'images');

%% 出力先作成
[~, ~] = mkdir(save_dir);
[~, ~] = mkdir(image_dir);

%% ランダムサンプリング
rng(rng_seed);
pitches = pitch_range(1) + rand(N, 1) * diff(pitch_range);
yaws    = yaw_range(1)   + rand(N, 1) * diff(yaw_range);
rolls   = roll_range(1)  + rand(N, 1) * diff(roll_range);

%% 初期化（1回だけ）
handles       = initBoxSim(cam_params.imageSize, target_size);
pose.distance = ref_distance;
pose.tx       = 0;
pose.ty       = 0;

%% レンダリングループ
csv_path = fullfile(save_dir, 'labels_euler.csv');
fid = fopen(csv_path, 'w');
fprintf(fid, 'image,pitch_deg,yaw_deg,roll_deg\n');

fprintf('=== ランダムデータベース生成 (%d枚) ===\n', N);
tic;

for i = 1:N
    pose.pitch = pitches(i);
    pose.yaw   = yaws(i);
    pose.roll  = rolls(i);

    img = renderBoxImage(handles, pose, cam_params);
    img = img > 0;  % 2値化

    rel_path = sprintf('images/sample_%06d.png', i);
    imwrite(img, fullfile(save_dir, rel_path));

    fprintf(fid, '%s,%.6f,%.6f,%.6f\n', rel_path, pitches(i), yaws(i), rolls(i));

    if mod(i, 1000) == 0
        fprintf('  %d/%d 完了 (%.1f秒)\n', i, N, toc);
    end
end

fclose(fid);
close(handles.fig);

fprintf('\n完了。%d枚 + ラベルを "%s/" に保存。(%.1f秒)\n', N, save_dir, toc);
fprintf('次のステップ: make_labels_csv.py でクォータニオンCSVに変換\n');
