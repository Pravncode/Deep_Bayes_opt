%% Optimized DE Antenna Optimization Script
clear;
clc;
tic;

% Initialize DE parameters
population = 100;         % Population size
maxIterations = 100;      % Number of generations
bounds = get_design_space_bounds();
nVars = size(bounds, 2);  % Number of variables
lb = bounds(1, :);
ub = bounds(2, :);

% Differential Evolution Parameters
F_min = 0.5;  % Minimum mutation factor
F_max = 1.0;  % Maximum mutation factor
CR = 0.7;     % Recombination (crossover) rate

% Generate Latin Hypercube samples
sample = lhsdesign(population, nVars);

% Scale samples to design bounds
scaled_sample = sample .* (ub - lb) + lb;

% Ensure valid design coordinates
design_samples = check_and_adjust_coordinates_within_bounds(scaled_sample, 25/2, 30/2);

% Initialize population
pop = design_samples;
fitness = zeros(population, 1);

% Evaluate initial population in parallel
parfor i = 1:population
    fitness(i) = objective_function(pop(i, :));
end

% Start Differential Evolution Loop
for iter = 1:maxIterations
    new_pop = pop;  % Placeholder for new population

    % Generate mutation factors (vectorized)
    F = F_min + rand(population, 1) * (F_max - F_min);  
    
    parfor i = 1:population
        % Select three distinct random indices
        r = randperm(population, 3);
        while any(r == i)
            r = randperm(population, 3);
        end

        x1 = pop(r(1), :);
        x2 = pop(r(2), :);
        x3 = pop(r(3), :);

        % Mutation (vectorized)
        mutant = x1 + F(i) * (x2 - x3);

        % Crossover (vectorized)
        cross_points = rand(1, nVars) < CR;
        if ~any(cross_points)  % Ensure at least one variable is changed
            cross_points(randi(nVars)) = true;
        end
        trial = pop(i, :);
        trial(cross_points) = mutant(cross_points);

        % Enforce bounds
        trial = max(lb, min(ub, trial));

        % Evaluate new candidate solution
        trial_fitness = objective_function(trial);

        % Selection: Replace if better
        if trial_fitness < fitness(i)
            new_pop(i, :) = trial;
            fitness(i) = trial_fitness;
        end
    end

    % Update population
    pop = new_pop;

    % Display progress
    disp(['Generation: ', num2str(iter), ', Best Fitness: ', num2str(min(fitness))]);
end

% Get the best solution
[best_fitness, best_idx] = min(fitness);
best_solution = pop(best_idx, :);

% Save final results
writematrix(best_solution, 'best_solution.txt', 'Delimiter', 'tab');
disp(['Best Fitness Value: ', num2str(best_fitness)]);
toc;

%% local functions

% Objective Function
function fitness = objective_function(design)
    %% Setup PCB and design processing parameters
    pcb_width = 25;   % PCB width in mm
    pcb_length = 30;  % PCB length in mm

    % Process the design (mirror, extract coordinates, remove redundant variables, etc.)
    design_samples = design;
    mirrored_sample = mirror_design_best(design_samples, pcb_width, pcb_length);
    [G1_original] = extract_G1_G2_coords(design_samples);
    [G1_mirrored] = extract_G1_G2_coords(mirrored_sample);
    design_samples_trimmed = remove_redundant_variables(design_samples);
    mirrored_sample_trimmed = remove_redundant_variables(mirrored_sample);
    
    % Combine all variables into one array (each row corresponds to a design sample)
    pcb_comb_test = [design_samples_trimmed, mirrored_sample_trimmed, G1_original, G1_mirrored];

    %% Run the antenna simulation to obtain S11 and gain vs. frequency
    [S11_data, gain_data] = antenna_simulations(pcb_comb_test);

    % If the simulation failed (e.g. returned all zeros) then assign a high penalty.
    if all(S11_data(:)==0) || all(gain_data(:)==0)
        fitness = 1e6;
        return;
    end

    %% Define target performance values
    target_S11 = -15;  % [dB] We wish S11 to be below -15 dB (i.e. more negative is better)
    target_gain = 3;   % [dBi] We wish the broadside gain to be at least 3 dBi

    %% Compute the penalties:
    % For S11: if the simulated S11 at a frequency is higher (less negative) than the target,
    % then add a penalty equal to (S11 - target). (If S11 is below the target, no penalty.)
    penalty_S11 = max(S11_data - target_S11, 0);  % size: [n_samples x n_freq]
    mean_penalty_S11 = mean(penalty_S11, 2);        % average over frequency for each sample

    % For gain: if the gain is below the target, add a penalty equal to (target - gain)
    penalty_gain = max(target_gain - gain_data, 0);
    mean_penalty_gain = mean(penalty_gain, 2);

    %% Combine the two penalties with equal weightage.
    % (Other normalizations/scalings could be used if needed.)
    total_penalty = 0.5 * mean_penalty_S11 + 0.5 * mean_penalty_gain;
    
    % If more than one sample is being evaluated, take the average penalty.
    fitness = mean(total_penalty);
end



% Define design space bounds
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
    g1_y_end_min = -5;
    g1_y_end_max = 10;

    lower_bounds = [
        lower_bounds, ...
        0.1 - epsilon, -half_width, -half_length, g1_y_end_min];
    upper_bounds = [
        upper_bounds, ...
        0.1 + epsilon, -half_width + epsilon, -half_length + epsilon, g1_y_end_max];

    bounds = [lower_bounds; upper_bounds];
end

function [G1] = extract_G1_G2_coords(design)
    % Extracts G1 and G2 coordinates from the design array.
    
    G1 = design(:, end-3:end);
end

function design = remove_redundant_variables(design)
    % Removes G1 and G2 coordinates from the design array.
    % Assumes G1 is at indices [-8:-4] and G2 is at indices [-4:].
    design(:, end-3:end) = [];
end

% Define the output function
function [x_opt, fval] = adaptive_DE(x, fval, lb, ub)
    pop_size = size(x, 1);
    nVars = size(x, 2);
    
    % DE parameters
    F_min = 0.5;
    F_max = 1.0;
    CR = 0.7; % Fixed recombination rate
    
    % Initialize trial population
    trial_pop = x;
    
    for i = 1:pop_size
        % Randomly select three distinct individuals
        r = randperm(pop_size, 3);
        while any(r == i)
            r = randperm(pop_size, 3);
        end
        
        x1 = x(r(1), :);
        x2 = x(r(2), :);
        x3 = x(r(3), :);
        
        % Adaptive Mutation Factor F
        F = F_min + rand * (F_max - F_min);
        
        % Perform mutation and crossover
        mutant = x1 + F * (x2 - x3);
        cross_points = rand(1, nVars) < CR;
        
        % Ensure at least one variable is updated
        if ~any(cross_points)
            cross_points(randi(nVars)) = true;
        end
        
        trial = x(i, :);
        trial(cross_points) = mutant(cross_points);
        
        % Clip to bounds
        trial = max(lb, min(ub, trial));
        
        % Evaluate the new solution
        trial_fitness = objective_function(trial);
        
        % Selection step
        if trial_fitness < fval(i)
            trial_pop(i, :) = trial;
            fval(i) = trial_fitness;
        end
    end
    
    % Return best solution
    [fval_best, best_idx] = min(fval);
    x_opt = trial_pop(best_idx, :);
end



% Mirror design along the y-axis and adjust coordinates
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

% Ensure coordinates are within bounds
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



% Antenna Simulations function
function [S11_data, gain_data] = antenna_simulations(pcb_comb_test)
    % Convert patch coordinates from mm to meters.
    patchCoords = pcb_comb_test * 1e-3;
    %disp('Input data size in antenna_simulations:');
    %disp(size(patchCoords));
    
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
            %% Build the antenna geometry for the i-th design sample
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
            
            %% Perform the electromagnetic simulation over the frequency sweep
            % Compute S-parameters; here we use a 50-Ohm port.
            S = sparameters(p, freq, 50);
            S11 = rfparam(S, 1, 1);
            S11_data(i, :) = 20 * log10(abs(S11));  % Convert to dB
            
            %% Compute the gain at the broadside direction (assumed theta = 0, phi = 0)
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

