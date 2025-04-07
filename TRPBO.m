% Uses RBF kernel and Expected Improvement (EI) acquisition function with Trust-Region Parallel Bayesian Optimization (TRPBO)
clear;
clc;
tic;

%% Parameters
pcb_width = 25;
pcb_length = 30;
half_width = pcb_width / 2;
half_length = pcb_length / 2;
num_iterations = 200;    % Number of BO iterations
q = 30;                 % Number of points to select per iteration
num_candidates = 5000;  % Number of random candidates to generate
input_dim = 56;
bounds = get_design_space_bounds();
lb = bounds(1, :);
ub = bounds(2, :);

% Initialize parallel pool (if not already initialized)
if isempty(gcp('nocreate'))
    parpool; % Start parallel pool
end

%% Load initial dataset
X_train = readmatrix('pcb_ini_200.txt'); 
y_ini = readmatrix('y_ini_200.txt'); 

y_train = zeros(size(y_ini, 1), 1);
for i = 1:size(y_ini, 1)
    S = y_ini(i, 1:100);
    G = y_ini(i, 101:200);
    y_train(i) = objective_function(S, G);
end

X_train = check_and_adjust_coordinates_within_bounds(X_train, half_width, half_length);
current_best = min(y_train);
fprintf('Initial Best Objective = %.4f\n', current_best);

%% Initialize Trust Region (TR)
x_center = mean(X_train, 1);
tr_radius = (ub - lb) / 2;

best_design = [];
best_objective = Inf;

% Open files for appending best results
best_obj_file = fopen('best_objectives.txt', 'a');
best_design_file = fopen('best_designs.txt', 'a');

for iter = 1:num_iterations
    % Train GP model with RBF kernel
    gp_model = fitrgp(X_train, y_train, 'KernelFunction', 'ardsquaredexponential', 'Standardize', true);
    
    % Generate random candidates within the Trust Region
    candidates = lhsdesign(num_candidates, input_dim);
    scaled_candidates = candidates .* (2 * tr_radius) + (x_center - tr_radius);
    adjusted_candidates = check_and_adjust_coordinates_within_bounds(scaled_candidates, half_width, half_length);
    
    % Calculate EI for all candidates
    [y_pred, y_sd] = predict(gp_model, adjusted_candidates);
    current_best = min(y_train);
    z = (current_best - y_pred) ./ max(y_sd, 1e-6);
    ei_values = (current_best - y_pred) .* normcdf(z) + y_sd .* normpdf(z);
    ei_values(y_sd == 0) = 0;
    
    % Select top q candidates with highest EI
    [~, sorted_indices] = sort(ei_values, 'descend');
    new_points = adjusted_candidates(sorted_indices(1:q), :);
    
    % Evaluate new candidates
    new_y = zeros(q, 1);
    parfor i = 1:q
        new_design = design_to_pcb_comb_test(new_points(i, :), pcb_width, pcb_length);
        [S_new, G_new] = antenna_simulations(new_design);
        new_y(i) = objective_function(S_new, G_new);
    end
    
    % Update dataset
    X_train = [X_train; new_points];
    y_train = [y_train; new_y];
    
    % Update Trust Region dynamically
    [current_best, best_idx] = min(y_train);
    if current_best < best_objective
        best_objective = current_best;
        best_design = X_train(best_idx, :);
        x_center = best_design;
        tr_radius = min(1.25 * tr_radius, ub - lb);
    else
        tr_radius = max(0.75 * tr_radius, (ub - lb) / 10);
    end
    
    % Append best results to files
    fprintf(best_obj_file, 'Iteration %d: %.4f\n', iter, current_best);
    fprintf(best_design_file, 'Iteration %d: %s\n', iter, mat2str(best_design));
    
    fprintf('Iteration %d: Best Objective = %.4f\n', iter, current_best);
end

% Close files
fclose(best_obj_file);
fclose(best_design_file);

writematrix(best_design, 'best_design.txt');
writematrix(best_objective, 'best_objective.txt');

fprintf('Optimization completed. Best Objective = %.4f\n', best_objective);
disp('Best design saved to best_design.txt');
disp('Best objective value saved to best_objective.txt');

toc;

%% Objective Function (Penalty-based)
function fitness = objective_function(S11_data, Gain_data)
    target_S11 = -15;   % S11 < -15 dB
    target_gain = 3;    % Gain > 3 dBi
    
    % Penalties (vectorized)
    S_penalty = mean(max(S11_data - target_S11, 0));
    G_penalty = mean(max(target_gain - Gain_data, 0));
    
    % Weighted sum
    fitness = 0.6 * S_penalty + 0.5 * G_penalty;
end

%% Expected Improvement Acquisition Function
function ei = expected_improvement(x, gp_model, best_y)
    [y_pred, y_sd] = predict(gp_model, x);
    z = (best_y - y_pred) ./ max(y_sd, 1e-6); % Avoid division by zero
    ei = (best_y - y_pred) .* normcdf(z) + y_sd .* normpdf(z);
    ei(y_sd == 0) = 0; % Handle points with zero uncertainty
end

%% MATLAB Simulation Function (Replace with your actual simulation)
function [S11_data, gain_data] = antenna_simulations(pcb_comb_test)
    % Convert patch coordinates from mm to meters.
    patchCoords = pcb_comb_test * 1e-3;
    disp('Input data size in antenna_simulations:');
    disp(size(patchCoords));
    
    % Number of design samples to simulate
    n_samples = size(patchCoords, 1);
    
    % Frequency sweep: 2 to 12 GHz (100 points)
    freq = linspace(2e9, 12e9, 100);
    
    % Substrate and PCB parameters
    er = 4.4;
    thick = 1.6e-3;
    pcb_width = 25.1e-3;   % in meters
    pcb_length = 30.1e-3;  % in meters
    
    % Initialize matrices to store results (one row per sample, one column per frequency)
    S11_data = zeros(n_samples, length(freq));
    gain_data = zeros(n_samples, length(freq));
    
    % Loop over all design samples (using parfor for speed if available)
    parfor i = 1:n_samples
        try
            % Build the antenna geometry for the i-th design sample
            % Start with a fixed metallic rectangle (for example, to set a feed area)
            fixed_metallic_area = antenna.Rectangle('Length', 2e-3, 'Width', 2e-3, 'Center', [0, -13.5e-3]);
            r = fixed_metallic_area;  % Radiating layer
            
            % Create the ground plane and an additional ground rectangle.
            GndPlane = antenna.Rectangle('Length', pcb_width, 'Width', pcb_length);
            Gnd = antenna.Rectangle('Length', pcb_width, 'Width', 2e-3, 'Center', [0, -13.5e-3]);
            
            % Add the radiating patch rectangles (assumed to be in the first 104 entries)
            radiating_coords = patchCoords(i, 1:104);
            for idx = 1:26  % Each rectangle is specified by 4 numbers (x_start, x_end, y_start, y_end)
                base_idx = (idx - 1) * 4 + 1;
                x_start = radiating_coords(base_idx);
                x_end = radiating_coords(base_idx + 1);
                y_start = radiating_coords(base_idx + 2);
                y_end = radiating_coords(base_idx + 3);
                
                dx = abs(x_end - x_start);
                dy = abs(y_end - y_start);
                cx = (x_start + x_end) / 2;
                cy = (y_start + y_end) / 2;
                
                % Ensure the rectangle is valid and within the PCB
                if dx > 0 && dy > 0 && cx >= -pcb_width/2 && cx <= pcb_width/2 && cy >= -pcb_length/2 && cy <= pcb_length/2
                    rect = antenna.Rectangle('Length', dx, 'Width', dy, 'Center', [cx, cy]);
                    r = r + rect;
                end
            end
            
            % Add ground patches (assumed to be given in entries 105 to 112)
            ground_coords = patchCoords(i, 105:112);
            for idx = 1:2  % Two ground rectangles in this example
                base_idx = (idx - 1) * 4 + 1;
                x_start_g = ground_coords(base_idx);
                x_end_g = ground_coords(base_idx + 1);
                y_start_g = ground_coords(base_idx + 2);
                y_end_g = ground_coords(base_idx + 3);
                
                dx_g = abs(x_end_g - x_start_g);
                dy_g = abs(y_end_g - y_start_g);
                cx_g = (x_start_g + x_end_g) / 2;
                cy_g = (y_start_g + y_end_g) / 2;
                
                if dx_g > 0 && dy_g > 0 && cx_g >= -pcb_width/2 && cx_g <= pcb_width/2 && cy_g >= -pcb_length/2 && cy_g <= pcb_length/2
                    g = antenna.Rectangle('Length', dx_g, 'Width', dy_g, 'Center', [cx_g, cy_g]);
                    Gnd = Gnd + g;
                end
            end
            
            % Define the substrate (using a dielectric material)
            d = dielectric('air');
            d.EpsilonR = er;
            d.Thickness = thick;
            
            % Create the PCB stack (the layers are radiating layer, dielectric, and ground)
            p = pcbStack;
            p.Name = 'Multilayer Antenna';
            p.BoardShape = GndPlane;
            p.BoardThickness = thick;
            p.Layers = {r, d, Gnd};
            
            % Specify the feed location (ensure it lies within the fixed metallic area)
            xfeed = 0;
            yfeed = -pcb_length/2 + 1e-3;
            p.FeedLocations = [xfeed, yfeed, 1, 3];
            p.FeedDiameter = 1e-3;
            
            % Perform the electromagnetic simulation over the frequency sweep
            % Compute S-parameters; here we use a 50-Ohm port.
            S = sparameters(p, freq, 50);
            S11 = rfparam(S, 1, 1);
            S11_data(i, :) = 20 * log10(abs(S11));  % Convert to dB
            
            % Compute the gain at the broadside direction (assumed theta = 0, phi = 0)
            gain_vals = zeros(1, length(freq));
            for j = 1:length(freq)
                % The pattern method returns the gain (in dBi) at the specified frequency and angles.
                gain_vals(j) = pattern(p, freq(j), 0, 90);
            end
            gain_data(i, :) = gain_vals;
            
            disp(['Iteration ', num2str(i), ' completed.']);
            
        catch ME
            disp(['Error in iteration ', num2str(i), ': ', ME.message]);
            % In case of an error for this sample, leave zeros.
            S11_data(i, :) = zeros(1, length(freq));
            gain_data(i, :) = zeros(1, length(freq));
        end
    end
    
    % (Optional) Save the simulation results to text files.
    writematrix(S11_data, 'S11_t.txt');
    writematrix(gain_data, 'Gain_t.txt');
end
%% extract G1
function [G1] = extract_G1_G2_coords(design)
       
    G1 = design(:, end-3:end);
end
%% remove redundant variable
function design = remove_redundant_variables(design)
    
    design(:, end-3:end) = [];
end

%% Define design space bounds
function bounds = get_design_space_bounds()
    pcb_width = 25;  % mm
    pcb_length = 30;  % mm
    half_width = pcb_width / 2;
    half_length = pcb_length / 2;

    grid_x_cells = 3;
    grid_y_cells = 4;
    patch_width_range = [2, 4];  % mm
    patch_length_range = [2, 7];  % mm
    offset_range = [-1, 1];  % mm

    lower_bounds = [];
    upper_bounds = [];
        

   

    % Adjust bounds slightly to avoid identical lower and upper bounds
    epsilon = 1e-2;

    lower_bounds = [
        0.1 - epsilon, -4, -half_length - epsilon, half_length - 1 - epsilon];
    upper_bounds = [
        0.1 + epsilon, -1, -half_length + epsilon, half_length - 1 + epsilon];

    for i = 1:(grid_x_cells * grid_y_cells)
        col = mod(i - 1, grid_x_cells);
        row = floor((i - 1) / grid_x_cells);
        cell_width = half_width / grid_x_cells;
        cell_height = pcb_length / grid_y_cells;
        center_x = -half_width + cell_width * (col + 0.5);
        center_y = -half_length + cell_height * (row + 0.5);

        lower_bounds = [
            lower_bounds, ...
            max(center_x + offset_range(1) - patch_width_range(2) / 2, -half_width), ...
            max(center_x + offset_range(1) + patch_width_range(1) / 2, -half_width), ...
            max(center_y + offset_range(1) - patch_length_range(2) / 2, -half_length), ...
            max(center_y + offset_range(1) + patch_length_range(1) / 2, -half_length)
        ];
        upper_bounds = [
            upper_bounds, ...
            min(center_x + offset_range(2) - patch_width_range(1) / 2, half_width), ...
            min(center_x + offset_range(2) + patch_width_range(2) / 2, half_width), ...
            min(center_y + offset_range(2) - patch_length_range(1) / 2, half_length), ...
            min(center_y + offset_range(2) + patch_length_range(2) / 2, half_length)
        ];
    end

    g1_y_start = -half_length;
    g1_y_end_min = -3;
    g1_y_end_max = 10;

    lower_bounds = [
        lower_bounds, ...
        0.1 - epsilon, -half_width, -half_length, g1_y_end_min];
    upper_bounds = [
        upper_bounds, ...
        0.1 + epsilon, -half_width + epsilon, -half_length + epsilon, g1_y_end_max];

    bounds = [lower_bounds; upper_bounds];
end



%% Mirror design along the y-axis and adjust coordinates
function mirrored_design = mirror_design_best(design, pcb_width, pcb_length)
    % Mirrors the design along the y-axis.
    half_width = pcb_width / 2;
    half_length = pcb_length / 2;

    % Check and adjust coordinates to ensure they are within bounds
    design = check_and_adjust_coordinates_within_bounds(design, half_width, half_length);

    mirrored_design = design;
    var_per_component = 4;  % x_start, x_end, y_start, y_end

    for i = 1:var_per_component:size(design, 2)
        x_start = design(:, i);
        x_end = design(:, i + 1);
        y_start = design(:, i + 2);
        y_end = design(:, i + 3);

        % Mirror the x-coordinates along the y-plane
        mirrored_x_start = -x_end;
        mirrored_x_end = -x_start;

        % Ensure x-coordinates are within bounds
        mirrored_x_start = min(max(mirrored_x_start, -half_width), half_width);
        mirrored_x_end = min(max(mirrored_x_end, -half_width), half_width);

        % Ensure y-coordinates are within bounds
        y_start = min(max(y_start, -half_length), half_length);
        y_end = min(max(y_end, -half_length), half_length);

        % Update mirrored_design
        mirrored_design(:, i:i + var_per_component - 1) = [mirrored_x_start, mirrored_x_end, y_start, y_end];
    end
end

%% Ensure coordinates are within bounds
function adjusted_coords = check_and_adjust_coordinates_within_bounds(coords, half_width, half_length)
    % Adjusts the coordinates to be within the specified bounds.
    adjusted_coords = coords;
    var_per_component = 4;  % x_start, x_end, y_start, y_end

    for i = 1:var_per_component:size(coords, 2)
        x_start = coords(:, i);
        x_end = coords(:, i + 1);
        y_start = coords(:, i + 2);
        y_end = coords(:, i + 3);

        % Adjust x coordinates
        x_start = min(max(x_start, -half_width), half_width);
        x_end = min(max(x_end, -half_width), half_width);

        % Adjust y coordinates
        y_start = min(max(y_start, -half_length), half_length);
        y_end = min(max(y_end, -half_length), half_length);

        adjusted_coords(:, i:i + var_per_component - 1) = [x_start, x_end, y_start, y_end];
    end
end
%% converting design into pcb_comb_test
function pcb_comb_test = design_to_pcb_comb_test(design, pcb_width, pcb_length)
    

    % Mirror the design
    mirrored_sample = mirror_design_best(design, pcb_width, pcb_length);
    
    % Extract G1 and G2 coordinates from original and mirrored designs
    G1_original = extract_G1_G2_coords(design);
    G1_mirrored = extract_G1_G2_coords(mirrored_sample);
    
    % Remove redundant variables from original and mirrored designs
    design_trimmed = remove_redundant_variables(design);
    mirrored_trimmed = remove_redundant_variables(mirrored_sample);
    
    % Combine all variables into one array
    pcb_comb_test = [design_trimmed, mirrored_trimmed, G1_original, G1_mirrored];
end