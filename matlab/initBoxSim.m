function handles = initBoxSim(imageSize, target_size)
% 直方体ターゲットのレンダリング環境を初期化する
% target_size: [幅(x), 高さ(y), 奥行(z)] (m)

w = target_size(1);
h = target_size(2);
d = target_size(3);

% 直方体の頂点（原点中心）
v = [-w/2, -h/2, -d/2;
      w/2, -h/2, -d/2;
      w/2,  h/2, -d/2;
     -w/2,  h/2, -d/2;
     -w/2, -h/2,  d/2;
      w/2, -h/2,  d/2;
      w/2,  h/2,  d/2;
     -w/2,  h/2,  d/2];

f = [1 2 3 4;   % -Z面
     5 6 7 8;   % +Z面
     1 2 6 5;   % -Y面
     2 3 7 6;   % +X面
     3 4 8 7;   % +Y面
     4 1 5 8];  % -X面

hFig = figure('Color', 'black', 'Visible', 'off', ...
              'Position', [100 100 imageSize imageSize]);
hAx  = axes('Parent', hFig, 'Color', 'black', 'Position', [0 0 1 1], ...
            'XColor', 'none', 'YColor', 'none', 'ZColor', 'none');
hold(hAx, 'on');

% 全6面を異なる色・異なるグレースケール輝度で設定（照明なし）
% 輝度 (0.299R + 0.587G + 0.114B): 白≈1.00, シアン≈0.70, 緑≈0.59,
%                                   マゼンタ≈0.41, 赤≈0.30, 青≈0.11
face_colors = [1.0 1.0 1.0;   % -Z: 白
               0.0 1.0 1.0;   % +Z: シアン
               0.0 1.0 0.0;   % -Y: 緑
               1.0 0.0 1.0;   % +X: マゼンタ
               1.0 0.0 0.0;   % +Y: 赤
               0.0 0.0 1.0];  % -X: 青

hBody = gobjects(6, 1);
for k = 1:6
    hBody(k) = patch('Parent', hAx, 'Vertices', v, 'Faces', f(k,:), ...
                     'FaceColor', face_colors(k,:), 'EdgeColor', 'none', ...
                     'FaceLighting', 'none');
end

axis(hAx, 'equal');
camproj(hAx, 'perspective');
camup(hAx, [0 1 0]);

lim = 300;
xlim(hAx, [-lim lim]);
ylim(hAx, [-lim lim]);
zlim(hAx, [-lim lim]);

handles = struct('fig', hFig, 'ax', hAx, 'body', hBody, ...
                 'v0', v, 'faces', f, 'imageSize', imageSize);
end
