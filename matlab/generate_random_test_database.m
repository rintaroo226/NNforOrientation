% generate_random_test_database.m
% generate_random_database.m と同一の手順でテスト用データベースを生成する。
% train (generate_random_database.m, rng_seed=42, N=10000) とは
% 別の乱数シードを使うことで、学習に使っていない姿勢のみで構成された
% ホールドアウトのテストセットにする。

addpath(fileparts(mfilename('fullpath')));  % initBoxSim, renderBoxImage

%% 設定
N            = 2000;        % サンプル数
rng_seed     = 999;         % train (42) とは異なるシード

cam_params.imageSize = 512;
cam_params.fov       = 25;
target_size          = [3, 2, 1];  % [幅, 高さ, 奥行] (m)  train と同一
ref_distance         = 10;

save_dir  = 'database_random_test';
image_dir = fullfile(save_dir, 'images');

%% 出力先作成
[~, ~] = mkdir(save_dir);
[~, ~] = mkdir(image_dir);

%% SO(3) 上で一様なクォータニオンをサンプリング（Shoemake の方法）
rng(rng_seed);
u1 = rand(N, 1);
u2 = rand(N, 1);
u3 = rand(N, 1);
qw = sqrt(1 - u1) .* sin(2 * pi * u2);
qx = sqrt(1 - u1) .* cos(2 * pi * u2);
qy = sqrt(u1)     .* sin(2 * pi * u3);
qz = sqrt(u1)     .* cos(2 * pi * u3);

%% 初期化（1回だけ）
handles       = initBoxSim(cam_params.imageSize, target_size);
pose.distance = ref_distance;
pose.tx       = 0;
pose.ty       = 0;

%% レンダリングループ
csv_path = fullfile(save_dir, 'labels.csv');
fid = fopen(csv_path, 'w');
fprintf(fid, 'image,qw,qx,qy,qz\n');

bin_threshold = 5;  % これより明るい画素を前景(255)とみなす（train と同一）

fprintf('=== テストデータベース生成 (%d枚) ===\n', N);
tic;

for i = 1:N
    q = [qw(i), qx(i), qy(i), qz(i)];
    [pose.pitch, pose.yaw, pose.roll] = quatToEulerZYX(q);

    img = renderBoxImage(handles, pose, cam_params);
    % 背景(黒)と前景(箱)のみの2値化シルエットとして保存
    bin_img = uint8((img > bin_threshold) * 255);

    rel_path = sprintf('images/sample_%06d.png', i);
    imwrite(bin_img, fullfile(save_dir, rel_path));

    fprintf(fid, '%s,%.8f,%.8f,%.8f,%.8f\n', rel_path, q(1), q(2), q(3), q(4));

    if mod(i, 1000) == 0
        fprintf('  %d/%d 完了 (%.1f秒)\n', i, N, toc);
    end
end

fclose(fid);
close(handles.fig);

fprintf('\n完了。%d枚 + ラベルを "%s/" に保存。(%.1f秒)\n', N, save_dir, toc);


function [pitch_deg, yaw_deg, roll_deg] = quatToEulerZYX(q)
% renderBoxImage.m の R = Rz(roll) * Ry(yaw) * Rx(pitch) 規約に合わせて
% クォータニオン [qw qx qy qz] を pitch/yaw/roll [deg] に変換する。
% yaw = ±90° 付近（ジンバルロック点）では pitch と roll の分解が
% 一意でなくなるが、レンダリング結果（回転行列）自体は正しいままなので
% 影響はない。
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
