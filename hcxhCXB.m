clear; clc; close all;
 
fs  = 1000;                 % 采样频率 (Hz)
dt  = 1/fs;                 % 采样间隔
T   = 1.0;                  % 信号总长 (s)
t   = (0:dt:T-dt).';        % 时间向量
N   = numel(t);

ricker = @(tt,t0,f) (1 - 2*(pi*f*(tt-t0)).^2) .* exp(-(pi*f*(tt-t0)).^2);

%% 2. 8 个子波参数： [t0(s), f(Hz), 极性*幅值]
params = [

   0.118,  60, +0.35;     % w1
    0.130,  65, -0.40;     % w2

    % --- 场景 2：频域重叠（低频、主频接近、时间紧邻）---
    0.305,  45, 0.75;     % w3 

    % --- 场景 3：高低频嵌套（低频载体 + 高频嵌入 + 末端大负峰）---
    0.520,  35, -0.30;     % w5  高频嵌入（小幅快振）
    0.540,  25, 0.50;     % w6  高频嵌入

    % --- 场景 4：时频二维逼近（3 子波，主频接近、时间紧邻）---
    0.725,  30, +0.60;     % w8
    0.755,  25, 0.20;     % w9
];
 
%% 叠加所有子波
s = zeros(N,1);
for k = 1:size(params,1)
    s = s + params(k,3) * ricker(t, params(k,1), params(k,2));
end

x = s(:);       % 转为列向量
dt = 1/fs;           % 采样间隔
Fs = fs;             % 采样频率
nsample = length(x);

% 可选：添加噪声
SNR_dB = 5;
signal_power = mean(x.^2);
noise_power = signal_power / (10^(SNR_dB/10));
rng(0);
noise = sqrt(noise_power) * randn(size(x));
x = x + noise;
%% === Step 2: 超小波参数设置 ===
fstart = 1;
fstop = 120;
delpha = 1;
f_num = fstart:delpha:fstop;
c1 = 1;
o = [1,3];
order_frac = linspace(o(1), o(2), numel(f_num));

%% === Step 3: 统一长度的超小波字典构建 ===
% 计算所有小波的最大长度，并统一到这个长度
max_wl_time = 0;
for i = 1:length(f_num)
    freq = f_num(i);
    n_wavelets = ceil(order_frac(i));
    for j = 1:n_wavelets
        n_cyc = j * c1;
        sd = (n_cyc / 2) * (1 / freq) / 2.5;
        wl_time = 6 * sd;
        max_wl_time = max(max_wl_time, wl_time);
    end
end

% 统一时间长度
t_length_unified = round(fs * max_wl_time);
if mod(t_length_unified, 2) == 0
    t_length_unified = t_length_unified + 1;
end

fprintf('\n统一字典长度: %d 个采样点\n', t_length_unified);

% 构建统一长度的字典矩阵
num_atoms = sum(ceil(order_frac));
source_dict = cell(1, num_atoms);  % 存储复数小波
all_freqs = zeros(1, num_atoms);
all_orders = zeros(1, num_atoms);

col_index = 0;
for i = 1:length(f_num)
    freq = f_num(i);
    n_wavelets = ceil(order_frac(i));
    
    for j = 1:n_wavelets
        n_cyc = j * c1;
        sd = (n_cyc / 2) * (1 / freq) / 2.5;
        wl_time = 6 * sd;
        
        % 生成原始小波（复数）
        t_length_local = round(fs * wl_time);
        if mod(t_length_local, 2) == 0
            t_length_local = t_length_local + 1;
        end
        t_local = linspace(-wl_time/2, wl_time/2, t_length_local);
        wavelet = bw_cf(t_local, sd, freq);  % 复Morlet小波
        wavelet = wavelet / max(abs(wavelet));
        
        % 补零到统一长度
        wavelet_unified = zeros(1, t_length_unified);
        start_idx = floor((t_length_unified - t_length_local) / 2) + 1;
        end_idx = min(start_idx + t_length_local - 1, t_length_unified);
        wavelet_unified(start_idx:end_idx) = wavelet(1:end_idx-start_idx+1);
        
        % 适配到 nsample 长度
        source = wavelet_unified;
        if length(source) ~= nsample
            if length(source) > nsample
                center = round(length(source)/2);
                start_idx_crop = center - floor(nsample/2);
                source = source(start_idx_crop:start_idx_crop+nsample-1);
            else
                padded_source = zeros(1, nsample);
                center = round(nsample/2);
                start_idx_pad = center - floor(length(source)/2);
                padded_source(start_idx_pad:start_idx_pad+length(source)-1) = source;
                source = padded_source;
            end
        end
        
        col_index = col_index + 1;
        source_dict{col_index} = source(:);  % 存储为列向量（复数）
        all_freqs(col_index) = freq;
        all_orders(col_index) = j;
    end
end

fprintf('字典原子总数: %d\n', num_atoms);

%% === Step 4: 构建复数卷积字典（使用Toeplitz矩阵方法）===
fprintf('\n构建复数卷积字典矩阵...\n');

% 显式声明为复数类型
Dict_full = zeros(nsample, nsample * num_atoms, 'like', 1i);

for atom_idx = 1:num_atoms
    source = source_dict{atom_idx};  % 复数小波
    T = conv_matrix_same(source, nsample);  % 保留复数
    start_col = (atom_idx-1)*nsample + 1;
    end_col = atom_idx*nsample;
    Dict_full(:, start_col:end_col) = T;
    
    if mod(atom_idx, 50) == 0
        fprintf('  构建进度: %d/%d\n', atom_idx, num_atoms);
    end
end
total_cols = size(Dict_full, 2);
for col = 1:total_cols
    col_norm = norm(Dict_full(:, col));
    if col_norm > 1e-10  % 避免除以零
        Dict_full(:, col) = Dict_full(:, col) / col_norm;
    end
    
    if mod(col, 10000) == 0

    end
end


%% === Step 5: L1-L2稀疏反演（单道）- 使用调整后的参数 ===
pm.lambda = 0.4;
pm.alpha = 0.6;
pm.delta = 1e5;
pm.maxit = 30;
pm.reltol = 1e-3;
tic;
[x_sparse_complex, ~] = CS_L1L2_uncon_ADMM(Dict_full, x, pm);
reshaped_coef = reshape(x_sparse_complex, nsample, num_atoms);
total_time = toc;

%% === Step 6: 分频重构（参照代码二的原理）===


% 预分配频率响应矩阵
freq_response_matrix = zeros(nsample, length(f_num));

for freq_idx = 1:length(f_num)

    target_indices = find(abs(all_freqs - f_num(freq_idx)) < 0.5);
    
    if isempty(target_indices)
        continue;
    end
    
    R_fi_full = zeros(nsample, 1);
    
    % 对当前频率的所有小波进行重构
    for k = 1:length(target_indices)
        j = target_indices(k);
        coef = reshaped_coef(:, j);  % 获取该小波的所有时间系数（复数）
        % 参照代码二：直接取系数的模（不做卷积重构）
        R_fi_full =coef+abs(coef);
    end    
    % 存储该频率的响应
    freq_response_matrix(:, freq_idx) = R_fi_full;
end

fprintf('分频重构完成\n');

%% === Step 7: 可视化（保持与代码二相同的格式）===
t_plot = (0:nsample-1) * dt;

% 创建图形（与代码二完全一致）
figure('Position', [100, 100, 500, 400], 'Color','w');
subplot(1, 4, 2);
imagesc(f_num, t_plot, abs(freq_response_matrix));
colormap(jet);
xlabel('Frequency (Hz)');
ylabel('Time (s)');
title('(d)');
xlim([0, 100]);
xticks([20, 60, 100]);
%% === Step 8: 定量评价指标 ===
TFR = abs(freq_response_matrix);   % 时频谱（幅值）

% ---------- 指标1：Rényi 3阶熵（时频聚集度，越小越聚集）----------
P = TFR.^2;
P = P / sum(P(:));
P_pos = P(P > 0);
renyi_entropy = (1/(1-3)) * log2(sum(P_pos.^3));

% ---------- 指标2：L1/L2 稀疏度（越接近0越稀疏）----------
sparsity_l1l2 = sum(TFR(:)) / (sqrt(sum(TFR(:).^2)) * sqrt(numel(TFR)));
% ---------- 指标2：Gini 指数（时频稀疏度，越大越稀疏/越聚集）----------
v = sort(abs(TFR(:)));               % 升序排列
v = v(v >= 0);
Nv = numel(v);
L1 = sum(v);
if L1 > 0
    k = (1:Nv).';
    gini_index = 1 - 2 * sum( (v ./ L1) .* ((Nv - k + 0.5) / Nv) );
else
    gini_index = 0;
end

fprintf('Gini 指数              : %.4f\n', gini_index);
fprintf('Rényi 3阶熵            : %.4f\n', renyi_entropy);
%% === 辅助函数：构建same模式的卷积矩阵（支持复数）===
function T = conv_matrix_same(kernel, n)

    kernel_len = length(kernel);
    half_len = floor(kernel_len / 2);
    
    % 初始化Toeplitz矩阵的列（根据kernel类型自动确定）
    if ~isreal(kernel)
        col = zeros(n, 1, 'like', 1i);
        row = zeros(1, n, 'like', 1i);
    else
        col = zeros(n, 1);
        row = zeros(1, n);
    end
    
    % 填充第一列
    for i = 1:kernel_len
        row_idx = i - half_len;
        if row_idx >= 1 && row_idx <= n
            col(row_idx) = kernel(i);
        end
    end
    
    % 填充第一行
    for i = 1:kernel_len
        col_idx = half_len - i + 2;
        if col_idx >= 1 && col_idx <= n
            row(col_idx) = kernel(i);
        end
    end
    
    % 构建Toeplitz矩阵
    T = toeplitz(col, row);
end

%% === 辅助函数：复Morlet小波 ===
function res = bw_cf(t, bw, cf)
    cnorm = 1 / (bw * sqrt(2 * pi));
    exp1 = cnorm * exp(-(t.^2) / (2 * bw^2));
    res = exp(2i * pi * cf * t) .* exp1;
end