% rebuild_labels_multidistance.m
% generate_random_database_multidistance.m の乱数サンプリングは
% rng_seed + N で完全に決定論的(rand()の出力は毎回同じ)なので、labels.csv を
% 誤って上書き・消失させても、画像ファイル自体さえ残っていれば
% 再レンダリング無しで復元できる。
%
% 既存の images/sample_XXXXXX.png を1〜N まで走査し、実際に存在する
% ファイルについてだけ、シードから再計算した qw,qx,qy,qz,distance を
% labels.csv に書き出す。
%
% 重要: N / rng_seed / dist_min / dist_max は
% generate_random_database_multidistance.m と必ず一致させること
% (これらが違うと、rand() の出力系列がズレて画像と対応しなくなる)。
%
% 実行中のレンダリング(generate_random_database_multidistance.m)と
% 同時に走らせないこと(labels.csv の書き込みが競合する)。

addpath(fileparts(mfilename('fullpath')));

%% 設定 (generate_random_database_multidistance.m と同一の値にすること)
N        = 100000;
rng_seed = 43;
dist_min = 9;
dist_max = 60;

save_dir  = 'database_random_multidistance_10';
image_dir = fullfile(save_dir, 'images');

%% 乱数列を再現 (generate_random_database_multidistance.m と全く同じ手順)
rng(rng_seed);
u1 = rand(N, 1);
u2 = rand(N, 1);
u3 = rand(N, 1);
qw = sqrt(1 - u1) .* sin(2 * pi * u2);
qx = sqrt(1 - u1) .* cos(2 * pi * u2);
qy = sqrt(u1)     .* sin(2 * pi * u3);
qz = sqrt(u1)     .* cos(2 * pi * u3);

log_dist  = log(dist_min) + (log(dist_max) - log(dist_min)) * rand(N, 1);
distances = exp(log_dist);

%% 実在する画像ファイルにだけラベルを書き出す
csv_path = fullfile(save_dir, 'labels.csv');
fid = fopen(csv_path, 'w');
fprintf(fid, 'image,qw,qx,qy,qz,distance\n');

n_found = 0;
fprintf('画像ファイルを走査中 (N=%d)...\n', N);
for i = 1:N
    rel_path = sprintf('images/sample_%06d.png', i);
    if isfile(fullfile(save_dir, rel_path))
        fprintf(fid, '%s,%.8f,%.8f,%.8f,%.8f,%.4f\n', ...
            rel_path, qw(i), qx(i), qy(i), qz(i), distances(i));
        n_found = n_found + 1;
    end
end
fclose(fid);

fprintf('%d/%d 枚の画像に対応するラベルを復元しました: %s\n', n_found, N, csv_path);
if n_found < N
    fprintf(['残り %d 枚は画像自体がまだ無いため、generate_random_database_multidistance.m ' ...
        '(再開対応版)をそのまま実行すれば続きから生成されます。\n'], N - n_found);
end
