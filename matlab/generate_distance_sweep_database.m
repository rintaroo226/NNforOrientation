% generate_distance_sweep_database.m
% 現行モデルは generate_random_database.m の ref_distance=10 固定で学習されている。
% 実運用では対象までの距離は様々なので、距離が変わったときに姿勢推定精度が
% どの程度落ちるかをまず実測するための評価用データセットを生成する。
%
% 同じ姿勢集合(N個)を複数の距離それぞれで描画することで、
% 「姿勢の違い」ではなく「距離の違い」だけの影響を切り分けられるようにする。
% initBoxSim/renderBoxImage は無改変で再利用する。

addpath(fileparts(mfilename('fullpath')));  % initBoxSim, renderBoxImage

%% 設定
N         = 200;                            % 距離1つあたりの姿勢サンプル数
rng_seed  = 777;                            % train(42)/test(999)とは異なるシード
% 学習時の distance=10 を挟んで、近距離〜遠距離をスイープする
distances = [5, 7, 10, 14, 20, 28, 40, 55];

cam_params.imageSize = 512;
cam_params.fov       = 25;
target_size          = [3, 2, 1];  % [幅, 高さ, 奥行] (m)  train と同一
bin_threshold         = 5;         % train と同一

save_dir  = 'database_distance_sweep';
image_dir = fullfile(save_dir, 'images');
[~, ~] = mkdir(save_dir);
[~, ~] = mkdir(image_dir);

%% 姿勢は距離間で共通にする（SO(3)一様サンプリング, Shoemakeの方法）
rng(rng_seed);
u1 = rand(N, 1);
u2 = rand(N, 1);
u3 = rand(N, 1);
qw = sqrt(1 - u1) .* sin(2 * pi * u2);
qx = sqrt(1 - u1) .* cos(2 * pi * u2);
qy = sqrt(u1)     .* sin(2 * pi * u3);
qz = sqrt(u1)     .* cos(2 * pi * u3);

%% 初期化（1回だけ）
handles = initBoxSim(cam_params.imageSize, target_size);
pose.tx = 0;
pose.ty = 0;

csv_path = fullfile(save_dir, 'labels.csv');
fid = fopen(csv_path, 'w');
fprintf(fid, 'image,qw,qx,qy,qz,distance\n');

fprintf('=== 距離スイープ評価用データセット生成 (%d姿勢 x %d距離 = %d枚) ===\n', ...
    N, numel(distances), N * numel(distances));
tic;

img_idx = 0;
for di = 1:numel(distances)
    pose.distance = distances(di);
    for i = 1:N
        q = [qw(i), qx(i), qy(i), qz(i)];
        [pose.pitch, pose.yaw, pose.roll] = quatToEulerZYX(q);

        img = renderBoxImage(handles, pose, cam_params);
        % 背景(黒)と前景(箱)のみの2値化シルエットとして保存（train/testと同一）
        bin_img = uint8((img > bin_threshold) * 255);

        img_idx = img_idx + 1;
        rel_path = sprintf('images/sample_%06d.png', img_idx);
        imwrite(bin_img, fullfile(save_dir, rel_path));

        fprintf(fid, '%s,%.8f,%.8f,%.8f,%.8f,%.4f\n', ...
            rel_path, q(1), q(2), q(3), q(4), distances(di));
    end
    fprintf('  距離=%.1f 完了 (累計%.1f秒)\n', distances(di), toc);
end

fclose(fid);
close(handles.fig);

fprintf('\n完了。%d枚 + ラベルを "%s/" に保存。(%.1f秒)\n', img_idx, save_dir, toc);


function [pitch_deg, yaw_deg, roll_deg] = quatToEulerZYX(q)
% generate_random_test_database.m と同一の変換。
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
