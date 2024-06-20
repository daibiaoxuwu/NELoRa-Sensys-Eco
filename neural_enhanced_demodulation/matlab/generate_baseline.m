clc;
clear;
close all;
addpath('Utils')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: Decode noisy chirp symbols with the baseline method: the dechirp
% Author: Chenning Li, Hanqing Guo
% Input: Noisy chirp symbols
% Output: The SNR-SER data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set your own paths
data_root = '';
raw_data_dir='path/to/raw_sf7_cross_instance';
% load settings
Fs = param_configs(3);         % sample rate
BW = param_configs(2);         % LoRa bandwidth
SF = param_configs(1);         % LoRa spreading factor
upsamping_factor = param_configs(4);         

nsamp = Fs * 2^SF / BW;
raw_data_list = dir(fullfile(raw_data_dir, '**\*.*'));
raw_data_list = raw_data_list(~[raw_data_list.isdir]); 
n_raw_data_list=length(raw_data_list);


SNR_list=[-25:-10,35];

chirp_down = Utils.gen_symbol(0,true);



% generate multi-path signal
Fs = param_configs(3);         % sample rate
upsamping_factor = param_configs(4);   

abs_decode = 0;



SNR_minimal=-30;
SNR_list=SNR_minimal:0;

BW_list=[125000];
SF_list=[7];
batch_list=4:7;

for BW=BW_list
    for SF=SF_list
        chirp_down = Utils.gen_symbol(0,true,Fs,BW,SF);
        error_matrix=zeros(length(SNR_list),1);
        error_matrix_count=zeros(length(SNR_list),1);
        nsamp = Fs * 2^SF / BW;
        for raw_data_index=1:n_raw_data_list

            raw_data_name=raw_data_list(raw_data_index);
            raw_data_name=fullfile(raw_data_name.folder, raw_data_name.name);
            [pathstr,raw_data_name_whole,ext] = fileparts(raw_data_name);
            raw_data_name_components = strsplit(raw_data_name_whole,'_');
            test_str=raw_data_name_components{1};
            if strcmp(test_str,'demod')==1||strcmp(test_str,'pt')==1
                continue;
            end
            [~,packet_index,~] = fileparts(pathstr);

            batch_index=str2num(raw_data_name_components{6});
            if (~ismember(batch_index, batch_list))
                    continue;
            end

            %% generate chirp symbol with code word (between [0,2^SF))
            chirp_raw = io_read_iq(raw_data_name);

            symbol_index=str2num(raw_data_name_components{1});
            
            %% conventional signal processing
            chirp_dechirp = chirp_raw .* chirp_down;
            chirp_fft_raw =(fft(chirp_dechirp, nsamp*upsamping_factor));

            % align_win_len = length(chirp_fft_raw) / (Fs/BW);   
            % chirp_fft_overlap=chirp_fft_raw(1:align_win_len)+chirp_fft_raw(end-align_win_len+1:end);
            % chirp_fft_overlap=flip(chirp_fft_overlap);
            % chirp_peak_overlap=abs(chirp_fft_overlap);
            % [pk_height_overlap,pk_index_overlap]=max(chirp_peak_overlap);

            chirp_peak_overlap=abs(chirp_abs_alias(chirp_fft_raw, Fs/BW));
            % chirp_peak_overlap = abs(chirp_comp_alias(chirp_fft_raw, Fs / BW));

            [pk_height_overlap,pk_index_overlap]=max(chirp_peak_overlap);
            code_estimated=mod(round(pk_index_overlap/upsamping_factor),2^SF);

            code_label=str2double(raw_data_name_components{2});
            code_label=mod(round(code_label),2^SF);
            for SNR=SNR_list
                if SNR ~=35
                    chirp = Utils.add_noise(chirp_raw, SNR);
                    SNR_index=SNR;
                else
                    chirp = chirp_raw;
                    SNR_index=35;
                end
                if (length(chirp)~=8*2^SF)
                    continue;
                end
                feature_data_name = [num2str(code_estimated),'_',num2str(SNR_index),'_',num2str(SF),'_',num2str(BW),'_',num2str(batch_index),'_',num2str(code_label),'_',num2str(packet_index),'_',num2str(symbol_index),'.mat'];
            
                if strcmp(feature_data_name,'.')==1||strcmp(feature_data_name,'..')==1
                    continue;
                end
                raw_data_name_components = strsplit(feature_data_name(1:end-4),'_');
                
                if ( ~ismember(str2num(raw_data_name_components{2}), SNR_list) || ~ismember(str2num(raw_data_name_components{5}), batch_list))
                    continue;
                end
                
                SNR_index=str2num(raw_data_name_components{2})-SNR_minimal+1;
                
                chirp_dechirp = chirp .* chirp_down;
                
                chirp_fft_raw =(fft(chirp_dechirp, nsamp*upsamping_factor));
                
                if abs_decode
                    chirp_peak_overlap=abs(chirp_abs_alias(chirp_fft_raw, Fs/BW));
                else
                    chirp_peak_overlap = abs(chirp_comp_alias(chirp_fft_raw, Fs / BW));
                end

                [pk_height_overlap,pk_index_overlap]=max(chirp_peak_overlap);
                code_estimated=mod(2^SF-round(pk_index_overlap/upsamping_factor),2^SF);
                
                code_label=str2num(raw_data_name_components{6});
                
                error_matrix(SNR_index,1)=  error_matrix(SNR_index,1)+(code_estimated==code_label);
                error_matrix_count(SNR_index,1)=error_matrix_count(SNR_index,1)+1;
            end
        end
        error_matrix=error_matrix./error_matrix_count;
        feature_path = [data_root, 'matlab/evaluation/','baseline_error_matrix_',num2str(SF),'_',num2str(BW),'.mat'];
        save(feature_path, 'error_matrix','SNR_list');
    end
end
