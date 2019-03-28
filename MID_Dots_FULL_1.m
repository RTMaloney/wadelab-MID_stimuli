function MID_Dots_FULL_1 (PresentControl)

% This is a self-contained demo function for displaying the FULL cue motion-in-depth
% stimulus (ie 'normal' random dot stereogram with both CD and IOVD components; dot lifetime
% the same as IOVD stimulus).
% Either the actual stimulus or the control stimulus (all dots have randomised phase of disparity added)
% can be displayed, depending on the flag 'PresentControl' (see below).
% Contrast of left and right eye dots can also be set independently using 'LeftEyeContrast' or 'RightEyeContrast'.
% This demo written for testing of amblyopic patients.
% The parameter 'PresentControl' determines whether to present the actual stimulus, or it's 'null' motion control.
% Set to false to display the motion stimulus (the default), or true to display the control.
% R Maloney, Jan 2017

% If the flag for the control stimulus is not parsed, set 
% null control to 'off'
if nargin < 1
    PresentControl = false;
end

%Define some of the display parameters:
PPD = 37; %At 57cm viewing distance, there are 37 pixels/deg on the Viewpixx (46.4 on the Propixx)

%If you're using the PROpixx or Viewpixx
UsingVP = true;
useHardwareStereo = false;

%Switch for whether you want the annulus superimposed over the dots:
DrawAnnulus = true;

%And if you want to draw the fixation cross/rings:
DrawRings = true;

%To independently switch off L or R eyes:
LeftEyeContrast = 1;
RightEyeContrast = 1;

%Choose the screen: it is usually the max screen no. available.
%Frustratingly, the Shuttle XPC (purchased June 2015) always seems to make the Vpixx display == 1. Not sure why, & can't seem to change it.
%So if we're on that machine, need to -1 from the screen number:
[~, CompName] = system('hostname'); %find out the computer name
if strncmpi(CompName, 'pspcawshuttle', length('pspcawshuttle')) ... %normal strcmp not working here, can't figure out why...
        && (length(Screen('Screens'))>1) %and there is more than 1 display connected...
    WhichScreen = max( Screen( 'Screens' ) )-1;
else
    WhichScreen = max( Screen( 'Screens' ) ); %should be right for any other machine!
end

screenRect = Screen('Rect',WhichScreen); %get the screen resolution.
centreX = screenRect(3)/2;
centreY = screenRect(4)/2;
RefreshRate = Screen('NominalFrameRate', WhichScreen);

jheapcl; %clear the java heap space.

% Define the dot texture, a square-shaped sheet of dots.
%Make the texture the same size as the height of the screen
imsize = screenRect(4);

%Define dot density in dots/(deg^2):
dot_dens_per_deg2 = 0.75; %2.273;

%compute number of dots in the total available area
num_dots = round(dot_dens_per_deg2 * (imsize/PPD)^2); %this many dots in the full dot field

%Just check whether num_dots is odd or even: (important for when contrast polarity is assigned)
%If odd, -1 to make it even
if mod(num_dots,2) %if odd, mod = 1
    num_dots = num_dots-1;
end

%specify dot size:
dot_sigma_in_degrees = 0.075; %0.1 for amblyopes; %size of SD of the dot profile in degs/vis angle
dot_sigma = dot_sigma_in_degrees * PPD; %sigma in pixels
dotsize = round(dot_sigma * 10); %make the dots some multiple of sigma
%NOTE: dotsize is simply half the length of the sides of a square patch of pixels that the dot profile is placed within.
%It is dot_sigma that really determines the size of the dots. Obviously, dotsize must be larger than dot_sigma.

%Define the minimum spacing between the PEAKS of the dot positions (in pixels):
SpatialJitter = round(0.5 * PPD); %+ dotsize/2;
%NOTE: add dot radius (ie dotsize/2) ensures the EDGES of the dots are separated by the minimum, but there may not be space in the matrix!

% Specify frequency:
frequency  = 1; % in Hz
%If the frequency of the sine wave is specified in Hz (ie cycles/sec)
%then other units of time must also be in SECONDS!!!
period = 1/frequency; %in sec
%the effective frame rate PER EYE: since each eye is stimulated on successive video frames
%Remember that the per-eye frame rate on the Viewpixx/PROpixx is 60 Hz
PeyeFR = RefreshRate/2; %Per eye f.r. is always half the absolute f.r.

%The no. of frames for a full cycle:
%Remember that each eye samples the motion vector at the same point, so this is in terms of per-eye frame rate
%(otherwise the different eyes would sample successive points on the trajectory/sine wave: one eye would lead the other)
FrmsFullCyc = round(PeyeFR*period); %must have integer no. of frames

%Set up dot indices to determine dot lifetime:
%We need to re-randomise 1/3 of all dots on every frame
%This ensures all dots have a lifetime of 3 frames (or 50 ms, with a per-eye frame rate of 60 Hz).
%So we set up the indices for (roughly) every 1/3 of the dots, leaving no dot unturned
%This is the same for every trial so only needs to be set once.
DotThirdIndices = round(linspace(1,num_dots,4));
DotThirdIndices = [DotThirdIndices(1:3)', DotThirdIndices(2:4)'];
DotThirdIndices(2:3,1) = DotThirdIndices(2:3,1)+1;

%%%%-------------------------------------------------------------------------%%%%
%           Determine the dot positions for the very first frame
%%%%-------------------------------------------------------------------------%%%%

CurrDot=1;
dot_pos = zeros(num_dots,2); %assign dot location matrix
while CurrDot <= num_dots
    
    if CurrDot == 1 %set random coordinates for very first dot
        dot_pos(CurrDot,:) = imsize.*rand(1,2);
        CurrDot = CurrDot+1;
    else
        %set the next dot's random position
        dot_pos(CurrDot,:) = imsize.*rand(1,2);
        %find the smallest distance (in pixels) between the current dot and any other existing dot
        idx = 1:CurrDot-1; %index each existing dot except the current one
        d = min((dot_pos(idx,1)-dot_pos(CurrDot,1)).^2 + (dot_pos(idx,2)-dot_pos(CurrDot,2)).^2);
        d = sqrt(d);
        
        %Now if that distance is smaller than the minimum permitted distance, re-randomise the dot coordinates
        %This will continue until (at least) the minimum distance is met
        if d < SpatialJitter
            dot_pos(CurrDot,:) = imsize.*rand(1,2);
        else %if that minimum distance is met, move on to the next dot
            CurrDot = CurrDot+1;
        end
    end
end

%%%%-------------------------------------------------------------------------%%%%
%       replicate dot_pos by the number of frames needed (to allocate memory)
%%%%-------------------------------------------------------------------------%%%%

% Only need to replicate once, because at this stage, both eyes' images are identical
% (disparity/lateral shift has not been added)

dot_pos = repmat(dot_pos,1,1,FrmsFullCyc); %for the left

%%%%-------------------------------------------------------------------------%%%%
%                   Determine dot lifetime
%%%%-------------------------------------------------------------------------%%%%

%Shift the position of a random 1/3 of dots. This effectively ends the lifetime of the dot and begins it anew somewhere else

%set dot positions for each frame in a full cycle
for m = 1:FrmsFullCyc
    
    %Make the dots the same position as the previous frame.
    %1/3 will then be moved.
    if m~=1
        dot_pos(:,:,m) = dot_pos(:,:,m-1);
    end
    
    Curr3rd = mod(m,3)+1; %tells us whether this is a 1st, 2nd or 3rd frame, & determines which 3rd of dots to change
    CurrRows = DotThirdIndices(Curr3rd,:);
    dot_pos(CurrRows(1):CurrRows(2),:,m) = nan; %make the third we want to change NaNs to put them out of the calculations
    CurrDot = CurrRows(1);
    
    while CurrDot <= CurrRows(2) %go through all dots in this 3rd
        
        %set the next dot's random position
        dot_pos(CurrDot,:,m) = imsize.*rand(1,2);
        %find the smallest distance (in pixels) between the current dot and any other existing dot
        %Index all the existing dots except the current one to do this
        %This means excluding all the dots currently unassigned (defined as NaNs).
        CurrentEmpty = find(isnan(dot_pos(:,1,m)))'; %all rows currently empty
        idx = setdiff(1:num_dots, [CurrDot, CurrentEmpty]);
        
        %Find the smallest distance between the current dot and any other dot
        d = min((dot_pos(idx,1,m)-dot_pos(CurrDot,1,m)).^2 + (dot_pos(idx,2,m)-dot_pos(CurrDot,2,m)).^2);
        d = sqrt(d);
        
        %Now if that distance is smaller than the minimum permitted distance, re-randomise the dot coordinates
        %This will continue until (at least) the minimum distance is met
        if d < SpatialJitter
            dot_pos(CurrDot,:,m) = imsize.*rand(1,2);
        else %if that minimum distance is met, move on to the next dot
            CurrDot = CurrDot+1;
        end
        
    end %end of loop across the dots in the current 3rd
    
end %end of loop across all frames in a full cycle

%adjust the x positions so they are in the centre of the screen. Do this for all frames in one go:
%If we don't do this they will be shifted off to the left of screen
AdjustXBy = (screenRect(3) - screenRect(4))/2; %shift X dot positions by this amount to centre them, since image size <screenRect(3)
dot_pos(:,1,:) = dot_pos(:,1,:) + AdjustXBy;
dot_posL = dot_pos; % Save these positions as the Left eye. It can be duplicated to produce the right eye just before the disparity is added; prior to presentation.

% Duplicate the left eye dot positions to make the right eye dot positions.
% At this stage all dots are identical for the two eyes (disparity has not been added yet).
dot_posR = dot_posL;

%make the dot profile:
x = (1-dotsize)/2:(dotsize-1)/2;
%x = x/PPD; %rescale x into vis angle
[x,y] = meshgrid(x,x);
y = -y;
[a,r] = cart2pol(x,y);

%This gives us a white-peaked dot (+ve contrast polarity)
env = exp(-r.^2/(2*dot_sigma.^2));%gaussian window
env = env./max(max(abs(env))); % normalize peak to +/- 1
env2 = -env; %make a negative version to give us a black dot (-ve contrast polarity)

%set up the raised cosine annular window.
%specify parameters for the annulus:
inrad = PPD * 1;% inner radius of annulus (in pixels), for fixation spot
outrad = PPD * 12/2; %outer radius of annulus (in pixels)
% define extent of spatial raised cosine at edge of aperture (in pixels)
cos_smooth = dotsize; %make it one dot size wide
%This should plonk the window in the middle of the matrix, which is what we want
imsize2 = imsize*2; %double the texture size
x0 = (imsize2+1)/2;
y0 = (imsize2+1)/2;
J = ones(imsize2);
for (ii=1:imsize2)
    for (jj=1:imsize2)
        r2 = (ii-x0)^2 + (jj-y0)^2;
        if (r2 > outrad^2)
            J(ii,jj) = 0;
        elseif (r2 < inrad^2)
            J(ii,jj) = 0;
        elseif (r2 > (outrad - cos_smooth)^2)
            J(ii,jj) = cos(pi.*(sqrt(r2)-outrad+cos_smooth)/(2*cos_smooth))^2;
        elseif (r2 < (inrad + cos_smooth)^2)
            J(ii,jj) = cos(pi.*(sqrt(r2)-inrad-cos_smooth)/(2*cos_smooth))^2;
        end
    end
end

%%%%-------------------------------%%%%
%       Set up fixation
%%%%-------------------------------%%%%

%Set up the fixation cross or spot:
%This is drawn directly to the screen using Screen('FillRect')
crossWidth = 2;
crossHeight = 10;
fixationCross = fixation_cross(crossWidth,crossHeight,centreX,centreY);

%Make the fixation lock ring:
% We have an inner one around fixation and an outer one right on the edge of screen.
% These could probably be defined as a single texture (rather than 2) but I think that will complicate matters with the alpha-blending settings.
% (they are complicated enough already)
ringRadiusInner = PPD*0.5;                % ring surrounding fixation
ringRadiusOuter = screenRect(4)/2;        % outer edge (radius) of the ring: the edge of the screen
ringWidthInner = ringRadiusInner - PPD/4; % 1/4 of a degree thick
ringWidthOuter = ringRadiusOuter - PPD/3; % 1/3 of a degree thick

%Make the ring. It's in a 2*2 checkerboard pattern:
fixationRing = double(checkerboard(screenRect(4)/2,1) > 0.5);
%Define the ring:
xx = (1-imsize)/2:(imsize-1)/2;
[xx,yy] = meshgrid(xx,xx);
[~,r] = cart2pol(xx,yy);
% make the alpha mask for the rings, inner and outer.
ring_alphaInner = ((r>ringWidthInner+1) & (r<ringRadiusInner-1)); % Make the alpha mask a tiny bit thinner than the ring itself.
ring_alphaOuter = ((r>ringWidthOuter+1) & (r<ringRadiusOuter-1));

%Unify the keyboard names in case we run this on a mac:
KbName('UnifyKeyNames')

try %Start a try/catch statement, in case something goes awry with the PTB functions
    
    %----------------------------
    % Set up the screen
    %----------------------------
    
    % initialization of the display
    AssertOpenGL;
    % Open PTB onscreen window: We request a 32 bit per colour component
    % floating point framebuffer if it supports alpha-blending. Otherwise
    % the system shall fall back to a 16 bit per colour component framebuffer:
    PsychImaging('PrepareConfiguration');
    PsychImaging('AddTask', 'General', 'FloatingPoint32BitIfPossible');
    %Set the color range to be normalised between 0 and 1 (rather than 0-255):
    PsychImaging('AddTask', 'General', 'NormalizedHighresColorRange', 1);
    %Initialise the Vpixx device:
    
    if UsingVP        % Enable DATAPixx blueline support, and VIEWPixx scanning backlight for optimal 3D
        
        PsychImaging('AddTask', 'General', 'UseDataPixx');
        Datapixx('Open');
        %The following commands are included in demos that apparently work for both the Viewpixx AND the PROpixx, though they seem specific to the Viewpixx...
        Datapixx('DisableVideoScanningBacklight');    % optionally, turn it off first, in case the refresh rate has changed since startup
        Datapixx('EnableVideoScanningBacklight');     % Only required if a VIEWPixx.
        Datapixx('EnableVideoStereoBlueline');
        Datapixx('SetVideoStereoVesaWaveform', 2);    % If driving NVIDIA glasses
        
        if Datapixx('IsViewpixx3D') %If it's the Viewpixx3D
            
            %Datapixx('EnableVideoLcd3D60Hz');
            Datapixx('DisableVideoLcd3D60Hz'); %actually, disabling seems better for reducing crosstalk...
            
            subjectData.DisplayType = 'Viewpixx3D'; %set aside the device type for reference
            Datapixx('RegWr');
            
        elseif Datapixx('IsPropixx') %if it's the Propixx DLP projector
            
            subjectData.DisplayType = 'PROpixx'; %set aside the device type for reference
            Datapixx('SetPropixxDlpSequenceProgram',0); %set to normal RGB video processing for driving the LEDs & DLP MMDs
            %Datapixx('RegWr');
            
            %Modify the per-eye crosstalk on the PROpixx.
            %Apparently this cross-talk correction only works when using RB3D video mode,
            %where the red/blue channels contain the left/right eye greyscale images (which we are not using).
            %Datapixx('SetPropixx3DCrosstalkLR', 1);
            %Datapixx('SetPropixx3DCrosstalkRL', 1);
            Datapixx('RegWrRd'); %seem to need to do this after setting 'SetPropixxDlpSequenceProgram' to 0
        end
    end
    %No clue what the RegWr and RegWrRd commands are all about, but they are in the demos, so I've included them.
    
    % Open an on screen (grey) window and configure the imaging pipeline
    %Info about the 'blueline' mechanism for synching to the 3D glasses:
    % There seems to be a blueline generation bug on some OpenGL systems.
    % SetStereoBlueLineSyncParameters(windowPtr, windowRect(4)) corrects the
    % bug on some systems, but breaks on other systems.
    % We'll just disable automatic blueline, and manually draw our own bluelines!
    if useHardwareStereo
        [win, windowRect] = PsychImaging('OpenWindow', WhichScreen, 0.5, [], [], [], 1); %flag of 1 engages stereomode
        SetStereoBlueLineSyncParameters(win, windowRect(4)+10);
    else
        [win, windowRect] = PsychImaging('OpenWindow', WhichScreen, 0.5);
    end
    
    %Define the 'blue line' parameters
    blueRectLeftOn   = [0,                 windowRect(4)-1, windowRect(3)/4,   windowRect(4)];
    blueRectLeftOff  = [windowRect(3)/4,   windowRect(4)-1, windowRect(3),     windowRect(4)];
    blueRectRightOn  = [0,                 windowRect(4)-1, windowRect(3)*3/4, windowRect(4)];
    blueRectRightOff = [windowRect(3)*3/4, windowRect(4)-1, windowRect(3),     windowRect(4)];
    
    %HideCursor;
    %Do the gamma correction. When using the Viewpixx/PROpixx through the PsychImaging pipeline, we shouldn't use
    %Screen(‘LoadNormalizedGamma’) (see http://www.jennyreadresearch.com/research/lab-set-up/datapixx/)
    %The PROpixx device should have a linear lUT built in, but we will add this here for completeness.
    %R_gamma =
    %G_gamma =
    %B_gamma =
    %PsychColorCorrection('SetEncodingGamma', win, [1/R_gamma, 1/G_gamma, 1/B_gamma]);
    %raise priority level:
    priorityLevel=MaxPriority(win); Priority(priorityLevel);
    %Query the screen refresh rate:
    ifi = Screen('GetFlipInterval',win); %in sec
    
    %Set the alpha-blending:
    %We want a linear superposition of the dots should they overlap:
    %Just like the Gabors in GarboriumDemo.m (see there for further info).
    Screen('BlendFunction', win, GL_SRC_ALPHA, GL_ONE);
    % We also want alpha-blending for smooth (anti-aliased) dots...
    %not sure how this will conflict with the above command
    %about the linear superposition of dots... but it doesn't seem to cause problems
    Screen('BlendFunction', win, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    %Make the textures for the dots, 1 for black and white
    Dots(1) = Screen('MakeTexture',win,env,[],[],2); %white dot
    Dots(2) = Screen('MakeTexture',win,env2,[],[],2); %black dot
    
    %Generate the annulus texture:
    AnnImages = 0.5.*ones(imsize2,imsize2,2); %specify RGB matrix of annulus, grey
    AnnImages(:,:,2) = 1.*J; %specify Alpha channel of annulus
    annulus = Screen('MakeTexture',win,AnnImages,[],[],2);
    
    % Generate the Inner ring/fixation lock texture:
    ringMat(:,:,1) = fixationRing;
    ringMat(:,:,2) = ring_alphaInner;
    fixationRingTextureInner = Screen('MakeTexture',win,ringMat,[],[],2);
    
    % Generate the Outer ring/fixation lock texture:
    ringMat(:,:,1) = fixationRing;
    ringMat(:,:,2) = ring_alphaOuter;
    fixationRingTextureOuter = Screen('MakeTexture',win,ringMat,[],[],2);
    
    % Preallocate array with destination rectangles:
    % This also defines initial dot locations
    % for the very first drawn stimulus frame:
    texrect = Screen('Rect', Dots(1));
    
    %%%%-------------------------------------------------------------------------%%%%
    %       Set up values of the sinusoidal change in horizontal dot disparity
    %%%%-------------------------------------------------------------------------%%%%
    
    %Fixed parameters:
    %If the frequency of the sine wave is specified in Hz (ie cycles/sec)
    %then other units of time must also be in SECONDS!!!
    frequency = 1.18; %1.18 Hz based on RM & JA thresholds; temp frequency of motion, in Hz. We want 1/2 cycle in 500 ms, so 1 Hz
    period = 1/frequency; %in sec
    angFreq = 2 * pi * frequency; %for the sine wave (the periodic sine wave is always some multiple of 2*pi)
    FULL_disparity = 90/60 * PPD; %90 arcmin based on RM & JA thresholds; disparity in pixels, defined in arcmin (akin to the amplitude of the sine wave)
    
    %Determined parameters:
    
    %Length of the sine wave:
    %The sine wave should not begin at 0, because then we get issues with it wrapping back to the zero point at both the
    %end of the cycle and the beginning of the next cycle.
    %So begin at the time (in sec), that comes just after zero; this will be the period divided by the no. of frames needed to give a full cycle.
    t = linspace(period/FrmsFullCyc, period, FrmsFullCyc); %One complete cycle, IN SEC, in steps of per-eye frames
    
    %Now make one full cycle of the sine wave, no matter the frequency
    %Of course faster frequencies must 'jump' through a full cycle in fewer frames (this is akin to 'undersampling')
    %Because our screen frame rate is always a fixed function of time, we can't change this (ie by increasing the screen frame rate)
    SineWave =  FULL_disparity * sin(angFreq * t); %IOVD_disparity * sin(angFreq * t)
    
    %assign dstRects matrix
    dstRectsL = zeros(4, num_dots,FrmsFullCyc);
    dstRectsR = zeros(4, num_dots,FrmsFullCyc);
    
    
    % *** index the sine wave, depending on whether it's the actual or control stimulus
    % Now we need to set up an index of the sine wave so each dot has disparity added at a different phase for the control.
    % We must index each point of the sine wave, which is the same as the frames in a full cycle.
    % Each dot should have the disparity added smoothly, although it will be at a random phase for each dot,
    % meaning the overall average disparity will be the same
    % (this bit added 2/11/16)
    
    if PresentControl % For the null-motion control stimulus
        for nd = 1:num_dots
            SineWvIdx(nd, :) = circshift(1:FrmsFullCyc, [0 ceil(FrmsFullCyc*rand)]); %random phase for each dot
        end
        % Now we have a row of indices for every single dot. These indices are for the disparity sine wave, each at a random phase.
        
    else  % For the actual FULL cue motion stimulus
        SineWvIdx = repmat(1:FrmsFullCyc, num_dots, 1); %simply replicate for each dot.
    end
    
    % Shift dot trajectory: add disparity
    for fr = 1:FrmsFullCyc
        
        % determine dot coordinates: remember, y position does not change: horiz disparity only
        % Update dot position according to sinsoidal trajectory on each (per-eye) frame
        % Left eye: -sin
        dstRectsL(:,:,fr) = CenterRectOnPoint(texrect, ... % the size of the dot texture
            dot_posL(:,1,fr) - SineWave(SineWvIdx(:,fr))', ...  % the x positions
            dot_posL(:,2,fr))';                            % the y positions
        % Right eye: +sin
        dstRectsR(:,:,fr) = CenterRectOnPoint(texrect, ...
            dot_posR(:,1,fr) + SineWave(SineWvIdx(:,fr))', ...
            dot_posR(:,2,fr))';
        
    end
    
    % Now set up the dot texture indices:
    %Left eye:
    DotsIdxL = [repmat(Dots(1),1,num_dots/2); repmat(Dots(2),1,num_dots/2)]; %Dots(1) = white; Dots(2)=black
    DotsIdxL = repmat(reshape(DotsIdxL,1,num_dots)', 1, FrmsFullCyc); % Replicate for each frame
    %Right eye:
    DotsIdxR = [repmat(Dots(1),1,num_dots/2); repmat(Dots(2),1,num_dots/2)]; %Dots(1) = white; Dots(2)=black
    DotsIdxR = repmat(reshape(DotsIdxR,1,num_dots)', 1, FrmsFullCyc); % Replicate for each frame
    
    f = 0; %this value increases with each iteration, but on a per eye basis only
    %And it determines the phase of the motion at the given frame (ie place in the cycle)
    missedFrames = 0;
    KbCheck();
    vbl = Screen('Flip',win); %sync vbl to start time
    while ~KbCheck
        
        % Select left-eye image buffer for drawing:
        if useHardwareStereo
            Screen('SelectStereoDrawBuffer', win, 0);
        end
        
        %%%%------------------------------------------------%%%%
        %               Draw left eye stimulus:
        %%%%------------------------------------------------%%%%
        
        %Draw dots:
        Screen('Blendfunction', win, GL_SRC_ALPHA, GL_ONE); %turn on alpha blending for lin superposition: see GarboriumDemo.m
        Screen('DrawTextures',win,DotsIdxL(:,mod(f,FrmsFullCyc)+1), ... % dot colour, now indexed each frame
            [],dstRectsL(:,:,mod(f,FrmsFullCyc)+1), ... % dot position
            [],[],LeftEyeContrast) %Final argument is contrast
        
        %Superimpose the annulus:
        if DrawAnnulus
            Screen('Blendfunction', win, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA);
            Screen('DrawTexture',win,annulus);
            Screen('Blendfunction', win, GL_ONE, GL_ZERO);
        end
        
        %Now draw the fixation ring/lock: requires some fancy tweaks of the alpha settings:
        if DrawRings
            
            Screen('BlendFunction', win, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); %need to flip the alpha around again (anti-aliasing)
            Screen('DrawTexture',win, fixationRingTextureInner);
            Screen('DrawTexture',win, fixationRingTextureOuter);        %Draw the black fixation cross:
            Screen('FillRect',win,[0 0 0],fixationCross);
        end
        
        Screen('BlendFunction', win, GL_ONE_MINUS_DST_ALPHA, GL_DST_ALPHA); %Done drawing so flip alpha back again
        Screen('BlendFunction', win, GL_ONE, GL_ZERO);
        
        %Draw blue lines:
        Screen('FillRect', win, [0, 0, 1], blueRectLeftOn);
        Screen('FillRect', win, [0, 0, 0], blueRectLeftOff);
        
        % Select right-eye image buffer for drawing:
        if useHardwareStereo
            Screen('SelectStereoDrawBuffer', win, 1);
        else %Not sure what would happen if this 'else' was actually true. I suspect something would go wrong, as stim would be presented twice.
            %But this is more or less how the Vpixx people do it in their 'DatapixxImagingStereoDemo'
            Screen('DrawingFinished', win);
            [vbl , ~ , ~, missed] = Screen('Flip', win, vbl + (ifi*0.5)); %, [], [], 1); %update display on next refresh (& provide deadline)
            
            if missed > 0
                missedFrames = missedFrames + 1;
            end
        end
        
        %%%%------------------------------------------------%%%%
        %               Draw right eye stimulus:
        %%%%------------------------------------------------%%%%
        
        %Draw dots:
        Screen('Blendfunction', win, GL_SRC_ALPHA, GL_ONE); %turn on alpha blending for lin superposition: see GarboriumDemo.m
        Screen('DrawTextures',win,DotsIdxR(:,mod(f,FrmsFullCyc)+1), ... % dot colour, now indexed each frame
            [],dstRectsR(:,:,mod(f,FrmsFullCyc)+1), ... % dot position.
            [],[],RightEyeContrast) %Final argument is contrast
        
        %Superimpose the annulus:
        if DrawAnnulus
            Screen('Blendfunction', win, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA);
            Screen('DrawTexture',win,annulus);
            Screen('Blendfunction', win, GL_ONE, GL_ZERO);
        end
        
        %Now draw the fixation ring/lock: requires some fancy tweaks of the alpha settings:
        if DrawRings
            Screen('BlendFunction', win, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); %need to flip the alpha around again (anti-aliasing)
            Screen('DrawTexture',win, fixationRingTextureInner);
            Screen('DrawTexture',win, fixationRingTextureOuter);        %Draw the fixation cross:
            Screen('FillRect',win,[0 0 0],fixationCross);
        end
        
        Screen('BlendFunction', win, GL_ONE_MINUS_DST_ALPHA, GL_DST_ALPHA); %Done drawing so flip alpha back again
        Screen('BlendFunction', win, GL_ONE, GL_ZERO);
        
        %Draw blue lines:
        Screen('FillRect', win, [0, 0, 1], blueRectRightOn);
        Screen('FillRect', win, [0, 0, 0], blueRectRightOff);
        
        Screen('DrawingFinished', win);
        
        [vbl , ~ , ~, missed] = Screen('Flip', win, vbl + (ifi*0.5)); %, [], [], 1); %update display on next refresh (& provide deadline)
        
        f = f+1 %increment counter for next frame
        
        %keep record of any missed frames:
        if missed > 0
            missedFrames = missedFrames + 1;
        end
        
    end
    f
    missedFrames
    
    %%%%------------------------------------------------%%%%
    %               Close down screen:
    %%%%------------------------------------------------%%%%
    %turn off the prioritisation:
    Priority( 0 ); %restore priority
    
    if UsingVP        % close down the ViewPixx or ProPixx
        Datapixx('DisableVideoScanningBacklight');
        if Datapixx('IsViewpixx3D')
            Datapixx('DisableVideoLcd3D60Hz');
        end
        Datapixx('RegWr');
        %Datapixx('Close'); %closing it here might cause it to crash?
    end
    
    %Close down the screen:
    Screen('CloseAll')
    
    Datapixx('Close'); %closing the Datapixx here (after closing the screen) might stop it from crashing
    
    %Bring back the mouse cursor:
    ShowCursor();
    
catch MException
    
    if UsingVP        % close down the ViewPixx or ProPixx
        Datapixx('DisableVideoScanningBacklight');
        if Datapixx('IsViewpixx3D')
            Datapixx('DisableVideoLcd3D60Hz');
        end
        Datapixx('RegWr');
        %Datapixx('Close'); %closing it here might cause it to crash?
    end
    
    %Close down the screen:
    Screen('CloseAll')
    Datapixx('Close'); %closing the Datapixx here (after closing the screen) might stop it from crashing
    
    rethrow (MException)
    psychrethrow(psychlasterror)
    
end %End of try/catch statement