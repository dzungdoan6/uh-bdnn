clear;

addpath(genpath('DeepNN_HASH'));
addpath(genpath('utils'));
addpath(genpath('minFunc'));
addpath(genpath('yael'));

%% CONFIGURE DATASET FOR EXPERIMENT
dataset = 'mnist'; % please change 'mnist' or 'cifar-10'
basedir = 'dataset/';
bit_evals = [8 16 24 32]; % number of bits you want to evaluate

%% LET'S GO
map_our_hash = [];
map_itq = [];
disp('============== RUNNING BINARY DEEP NEURAL NETWORK ===============');
for bit = bit_evals
    fprintf(2,'USING %d BITS\n', bit);
    map = runHash( dataset, basedir, bit );
    map_our_hash = [map_our_hash ; map];
end
disp('================ RUNNING ITQ FOR COMPARISON ======================');
for bit = bit_evals
    map = runMainITQ( dataset, basedir, bit );
    map_itq = [map_itq ; map];
end

%% VISUALIZATION
interval = [0 8 16 24];
hold on;
font = 15;
set(gca, 'FontSize', font);
xlim([0 24]);
plot(interval, map_our_hash, 'Color', 'red', 'Marker', 's', 'Linewidth', 2);
hold on;
plot(interval, map_itq, 'Color', 'blue', 'Marker', 'o', 'Linewidth', 2);

lngd = legend('UH-BDNN', 'ITQ');
set(lngd, 'Location', 'southeast');
set(lngd, 'interpreter', 'latex');
set(lngd, 'fontsize', 15);
grid on;
xlabel('number of bits', 'fontsize', 20);
ylabel('mAP','fontsize', 20);
set(gca,'XTick',0:8:24);
set(gca,'XTickLabel',{'8','16','24','32'});
