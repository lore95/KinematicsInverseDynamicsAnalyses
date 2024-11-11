%%% Basic read data and visualization for the course SM2501
%%% Biomechanics of Human Movement at KTH - 2024
%% Setup files
name_motion={'Walking'    'Jogging'   'Crouch'};
name_grf   ={'Walking_FP' 'Jogging_FP' 'Crouch_FP'};

% frame_sel  =[];

index=3; % select the motion to be loaded and visualized e.g., index=1 -> NormWalk
%% Read marker trajectory and ground reaction data
% data files should be in the same folder as the .m file
file_dir = pwd;
data_trc = readtable(fullfile(file_dir,[name_motion{index} '.csv']));
data_grf = readtable(fullfile(file_dir,[name_grf{index} '.csv']));

%% Downsample ground reaction data
% down sample the ground reaction data, so it has the same length as marker trajectory
data_grf_s = downsample(data_grf,10);

%% Assign the uploaded table to variables in MATLAB
toMeters=1/1000; % data is originally in mm, it has to be divided by 1000 to have it in meters

RTOE_x=data_trc.RTOO_Y*toMeters;   RTOE_y=data_trc.RTOO_Z*toMeters;    % horizontal coordinate of the right toe & vertical coordinate of the right toe 
LTOE_x=data_trc.LTOO_Y*toMeters;   LTOE_y=data_trc.LTOO_Z*toMeters;    % horizontal coordinate of the left toe & vertical coordinate of the left toe 

RANKLE_x=data_trc.RAJC_Y*toMeters;   RANKLE_y=data_trc.RAJC_Z*toMeters;   
LANKLE_x=data_trc.LAJC_Y*toMeters;   LANKLE_y=data_trc.LAJC_Z*toMeters;   

RKNEE_x=data_trc.RKJC_Y*toMeters;   RKNEE_y=data_trc.RKJC_Z*toMeters;    
LKNEE_x=data_trc.LKJC_Y*toMeters;   LKNEE_y=data_trc.LKJC_Z*toMeters;   

RHIP_x=data_trc.RHJC_Y*toMeters;   RHIP_y=data_trc.RHJC_Z*toMeters;    
LHIP_x=data_trc.LHJC_Y*toMeters;   LHIP_y=data_trc.LHJC_Z*toMeters;  

PELO_x=data_trc.PELO_Y*toMeters;   PELO_y=data_trc.PELO_Z*toMeters;    
PELP_x=data_trc.PELP_Y*toMeters;   PELP_y=data_trc.PELP_Z*toMeters;   

TRXO_x=data_trc.TRXO_Y*toMeters;   TRXO_y=data_trc.TRXO_Z*toMeters;    
TRXP_x=data_trc.TRXP_Y*toMeters;   TRXP_y=data_trc.TRXP_Z*toMeters;  

FP1_force_x=data_grf_s.FP1_Force_Y;            FP1_force_y=data_grf_s.FP1_Force_Z;
FP1_COP_x  =data_grf_s.FP1_COP_Y*toMeters;     FP1_COP_y  =data_grf_s.FP1_COP_Z*toMeters;

FP2_force_x=data_grf_s.FP2_Force_Y;            FP2_force_y=data_grf_s.FP2_Force_Z;
FP2_COP_x=data_grf_s.FP2_COP_Y*toMeters;       FP2_COP_y=data_grf_s.FP2_COP_Z*toMeters;
%% Run a simple animation for visualization purposes only

run_animation=true;
color_body="#0077be";
row=height(data_trc);

if run_animation==true
figure('color','w','Position', [10 10 1200 700]);

    % Start from frame 50th as the coordinates at the first few frames were
    % not processed and they are not considered in the assignment
    for i=1:5:row   
    subplot(1,1,1)
    plot([RANKLE_x(i),RTOE_x(i)],[RANKLE_y(i),RTOE_y(i)],'Color',color_body,'LineWidth',2) % creates a segment between the ankle and the toe in the right side
    hold on
    plot([RANKLE_x(i),RKNEE_x(i)],[RANKLE_y(i),RKNEE_y(i)],'Color','g','LineWidth',2)
    plot([LANKLE_x(i),LTOE_x(i)],[LANKLE_y(i),LTOE_y(i)],'Color',color_body,'LineWidth',2)
    plot([LANKLE_x(i),LKNEE_x(i)],[LANKLE_y(i),LKNEE_y(i)],'Color','r','LineWidth',2)

    plot([LHIP_x(i),LKNEE_x(i)],[LHIP_y(i),LKNEE_y(i)],'Color',color_body,'LineWidth',2)
    plot([RHIP_x(i),RKNEE_x(i)],[RHIP_y(i),RKNEE_y(i)],'Color',color_body,'LineWidth',2)
    
    plot([PELO_x(i),PELP_x(i)],[PELO_y(i),PELP_y(i)],'Color',color_body,'LineWidth',2)
    plot([TRXO_x(i),TRXP_x(i)],[TRXO_y(i),TRXP_y(i)],'Color',color_body,'LineWidth',2)

    text(RANKLE_x(i),RANKLE_y(i),'Rankle','FontSize',8)
    text(LANKLE_x(i),LANKLE_y(i),'Lankle','FontSize',8)
    text(RKNEE_x(i),RKNEE_y(i),'Rknee','FontSize',8)
    text(LKNEE_x(i),LKNEE_y(i),'Lknee','FontSize',8)
    
    % Compare with the previous frame, if the values of COP are changed, it
    % means the subject is stepping on that force plate and we visualize it 
    if (i>=2)
        if FP1_COP_x(i) ~= FP1_COP_x(i-1) 
            plot(FP1_COP_x(i),FP1_COP_y(i),'ok','LineWidth',2);
            quiver(FP1_COP_x(i),FP1_COP_y(i),FP1_force_x(i)/1000,FP1_force_y(i)/1000,0,'Color','r','LineWidth',2); % It is divided by 1000 for visualization purpose only
        end
        if FP2_COP_x(i) ~= FP2_COP_x(i-1)
            plot(FP2_COP_x(i),FP2_COP_y(i),'ok','LineWidth',2);
            quiver(FP2_COP_x(i),FP2_COP_y(i),FP2_force_x(i)/1000,FP2_force_y(i)/1000,0,'Color','r','LineWidth',2); % It is divided by 1000 for visualization purpose only
        end
    end

    hold off
    axis([-3.5 3.5 -0.25 3])
    xlabel("horizontal distance [m]",'FontSize',20); ylabel("vertical distance [m]",'FontSize',20)
    pause(0.1)
    end
end
