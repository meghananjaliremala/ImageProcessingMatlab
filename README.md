# ImageProcessingMatlab
Matlab Image Processing LiveScript
```Matlab
% Read MRI and CT images
mri_image1 = imread('MRI_of_BRAIN2.jpg');
mri_image2 = imread('MRI_of_Human_Brain1.jpg');
ct_image1 = imread('CT-CHEST.jpg');
ct_image2 = imread('CT_Head (1).jpg');


% Blur the images
sigma = 2; % standard deviation for Gaussian blur
blurred_mri1 = imgaussfilt(mri_image1, sigma);
blurred_mri2 = imgaussfilt(mri_image2, sigma);
blurred_ct1 = imgaussfilt(ct_image1, sigma);
blurred_ct2 = imgaussfilt(ct_image2, sigma);

% Display the blurred images
figure;
subplot(2,2,1);
imshow(blurred_ct2);
title('Blurred CT Head');
subplot(2,2,2);
imshow(blurred_ct1);
title('Blurred CT Chest');
subplot(2,2,3);
imshow(blurred_mri1);
title('Blurred MRI of Brain 2');
subplot(2,2,4);
imshow(blurred_mri2);
title('Blurred MRI of Brain 1');

% Deblur the images using Wiener deconvolution
psf_size = 15; % PSF size
noise_var = 0.0001; % Estimated noise variance
deblurred_mri1_wiener = deconvwnr(blurred_mri1, fspecial('gaussian', [5,5], 2), 0.01);
deblurred_mri2_wiener = deconvwnr(blurred_mri2, fspecial('gaussian', [5,5], 2), 0.01);
deblurred_ct1_wiener = deconvwnr(blurred_ct1, fspecial('gaussian', [5,5], 2), 0.01);
deblurred_ct2_wiener = deconvwnr(blurred_ct2, fspecial('gaussian', [5,5], 2), 0.01);

% Deblur the images using Lucy-Richardson deconvolution
deblurred_mri1_lucy = deconvlucy(blurred_mri1, fspecial('gaussian', [5,5], 2), 10);
deblurred_mri2_lucy = deconvlucy(blurred_mri2, fspecial('gaussian', [5,5], 2), 10);
deblurred_ct1_lucy = deconvlucy(blurred_ct1, fspecial('gaussian', [5,5], 2), 10);
deblurred_ct2_lucy = deconvlucy(blurred_ct2, fspecial('gaussian', [5,5], 2), 10);

% Calculate Mean Square Error (MSE)
mse_mri1_wiener = immse(mri_image1, deblurred_mri1_wiener);
mse_mri2_wiener = immse(mri_image2, deblurred_mri2_wiener);
mse_ct1_wiener = immse(ct_image1, deblurred_ct1_wiener);
mse_ct2_wiener = immse(ct_image2, deblurred_ct2_wiener);

mse_mri1_lucy = immse(mri_image1, deblurred_mri1_lucy);
mse_mri2_lucy = immse(mri_image2, deblurred_mri2_lucy);
mse_ct1_lucy = immse(ct_image1, deblurred_ct1_lucy);
mse_ct2_lucy = immse(ct_image2, deblurred_ct2_lucy);

% Display MSE results
disp('Mean Square Error (Wiener Deconvolution):');
disp(['MRI Image 1: ', num2str(mse_mri1_wiener)]);
disp(['MRI Image 2: ', num2str(mse_mri2_wiener)]);
disp(['CT Image 1: ', num2str(mse_ct1_wiener)]);
disp(['CT Image 2: ', num2str(mse_ct2_wiener)]);
disp('Mean Square Error (Lucy-Richardson Deconvolution):');
disp(['MRI Image 1: ', num2str(mse_mri1_lucy)]);
disp(['MRI Image 2: ', num2str(mse_mri2_lucy)]);
disp(['CT Image 1: ', num2str(mse_ct1_lucy)]);
disp(['CT Image 2: ', num2str(mse_ct2_lucy)]);

% Plot Normalized Error Distribution
figure;
subplot(2,2,1);
histogram((double(mri_image1) - double(deblurred_mri1_wiener)).^2, 'Normalization', 'pdf');
title('MRI Image 1 - Wiener Deconvolution');
subplot(2,2,2);
histogram((double(mri_image1) - double(deblurred_mri1_lucy)).^2, 'Normalization', 'pdf');
title('MRI Image 1 - Lucy-Richardson Deconvolution');
subplot(2,2,3);
histogram((double(mri_image2) - double(deblurred_mri2_wiener)).^2, 'Normalization', 'pdf');
title('MRI Image 2 - Wiener Deconvolution');
subplot(2,2,4);
histogram((double(mri_image2) - double(deblurred_mri2_lucy)).^2, 'Normalization', 'pdf');
title('MRI Image 2 - Lucy-Richardson Deconvolution');

% Display estimated PSF and noise variance
estimated_psf = fspecial('gaussian', psf_size);
fprintf('Estimated PSF:\n');
disp(estimated_psf);
fprintf('Estimated Noise Variance: %f\n', noise_var);

% Plot PSF
figure;
surf(estimated_psf);
title('Estimated PSF');
xlabel('X-axis');
ylabel('Y-axis');




% Compare images to original
figure;
subplot(2,2,1);
histogram((double(ct_image1) - double(deblurred_ct1_wiener)).^2, 'Normalization', 'pdf');
title('CT Image 1 - Wiener Deconvolution');
subplot(2,2,2);
histogram((double(ct_image1) - double(deblurred_ct1_lucy)).^2, 'Normalization', 'pdf');
title('CT Image 1 - Lucy-Richardson Deconvolution');
subplot(2,2,3);
histogram((double(ct_image2) - double(deblurred_ct2_wiener)).^2, 'Normalization', 'pdf');
title('CT Image 2 - Wiener Deconvolution');
subplot(2,2,4);
histogram((double(ct_image2) - double(deblurred_ct2_lucy)).^2, 'Normalization', 'pdf');
title('CT Image 2 - Lucy-Richardson Deconvolution');

% Display the filtered images
figure;
subplot(2,2,1);
imshow(blurred_ct2);
title('Blurred CT Head');
subplot(2,2,2);
imshow(deblurred_ct2_wiener);
title('Wiener Filtered CT Head');
subplot(2,2,3);
imshow(deblurred_ct2_lucy);
title('Lucy Filtered CT Head');

figure;
subplot(2,2,4);
imshow(blurred_ct1);
title('Blurred CT Chest');
subplot(2,2,1);
imshow(deblurred_ct1_wiener);
title('Wiener Filtered CT Chest');
subplot(2,2,2);
imshow(deblurred_ct1_lucy);
title('Lucy Filtered CT Chest');

figure;
subplot(2,2,1);
imshow(blurred_mri1);
title('Blurred MRI of Brain 2');
subplot(2,2,2);
imshow(deblurred_mri1_wiener);
title('Wiener Filtered MRI of Brain 2');
subplot(2,2,3);
imshow(deblurred_mri1_lucy);
title('Lucy Filtered MRI of Brain 2');

figure;
subplot(2,2,4);
imshow(blurred_mri2);
title('Blurred MRI of Brain 1');
subplot(2,2,1);
imshow(deblurred_mri2_wiener);
title('Wiener Filtered MRI of Brain 1');
subplot(2,2,2);
imshow(deblurred_mri2_lucy);
title('Lucy Filtered MRI of Brain 1')




% Read MRI and CT Lucy images
mri_image1 = imread('lucy_filtered_mribrain2.jpg');
mri_image2 = imread('lucy_filtered_mribrain1.jpg');
ct_image1 = imread('lucy_filtered_ctchest.jpg');
ct_image2 = imread('lucy_filtered_cthead.jpg');

% Store all images in a cell array
images = {mri_image1, mri_image2, ct_image1, ct_image2};

% Initialize cell array to store denoised images
denoised_images = cell(size(images));

% Perform wavelet denoising on each image
for i = 1:numel(images)
    % Convert to grayscale if necessary
    if size(images{i}, 3) == 3
        images{i} = rgb2gray(images{i});
    end

    % Perform Wavelet Denoising
    % 'sym4' is the chosen wavelet, and you can experiment with other wavelets
    % The level of decomposition is set to 1, but you can increase it as needed
    [denoisedImage,~] = wdenoise(double(images{i}), 1, 'Wavelet', 'sym4');

    % Convert back to uint8 if necessary
    denoised_images{i} = uint8(denoisedImage);
end

% Display the original and denoised images
figure;
for i = 1:numel(images)
    subplot(2, numel(images), i);
    imshow(images{i});
    title(['Original Image ', num2str(i)]);
   
    subplot(2, numel(images), i + numel(images));
    imshow(denoised_images{i});
    title(['Denoised Image ', num2str(i)]);
end


% Color Image Generation for MRI
% Convert to grayscale if necessary
if size(mri_image1, 3) == 3
    mri_image1 = rgb2gray(mri_image1);
end

% Normalize the image intensity to range [0, 1]
mri_image1 = mat2gray(mri_image1);

% Define intensity thresholds
lowThreshold = 0.33;
midThreshold = 0.66;

% Assign colors based on intensity levels
% Low intensity: Blue
% Medium intensity: Green
% High intensity: Red
coloredImage_mri1 = zeros([size(mri_image1), 3]); % Initialize a color image
coloredImage_mri1(:,:,3) = (mri_image1 <= lowThreshold); % Blue channel
coloredImage_mri1(:,:,2) = (mri_image1 > lowThreshold) & (mri_image1 <= midThreshold); % Green channel
coloredImage_mri1(:,:,1) = (mri_image1 > midThreshold); % Red channel

% Display the color image for MRI
figure;
imshow(coloredImage_mri1);
title('Intensity-Based Colored MRI Image (MRI of Brain 2)');

% Color Image Generation for MRI
% Convert to grayscale if necessary
if size(mri_image2, 3) == 3
    mri_image2 = rgb2gray(mri_image2);
end

% Normalize the image intensity to range [0, 1]
mri_image2 = mat2gray(mri_image2);

% Define intensity thresholds
lowThreshold = 0.33;
midThreshold = 0.66;

% Assign colors based on intensity levels
% Low intensity: Blue
% Medium intensity: Green
% High intensity: Red
coloredImage_mri2 = zeros([size(mri_image2), 3]); % Initialize a color image
coloredImage_mri2(:,:,3) = (mri_image2 <= lowThreshold); % Blue channel
coloredImage_mri2(:,:,2) = (mri_image2 > lowThreshold) & (mri_image2 <= midThreshold); % Green channel
coloredImage_mri2(:,:,1) = (mri_image2 > midThreshold); % Red channel

% Display the color image for MRI
figure;
imshow(coloredImage_mri2);
title('Intensity-Based Colored MRI Image (MRI of Brain 2)');




% Color Image Generation for CT
% Convert to grayscale if necessary
if size(ct_image1, 3) == 3
    ct_image1 = rgb2gray(ct_image1);
end

% Normalize the image intensity to range [0, 1]
ct_image1 = mat2gray(ct_image1);

% Assign colors based on intensity levels
% Low intensity: Blue
% Medium intensity: Green
% High intensity: Red
coloredImage_ct1 = zeros([size(ct_image1), 3]); % Initialize a color image
coloredImage_ct1(:,:,3) = (ct_image1 <= lowThreshold); % Blue channel
coloredImage_ct1(:,:,2) = (ct_image1 > lowThreshold) & (ct_image1 <= midThreshold); % Green channel
coloredImage_ct1(:,:,1) = (ct_image1 > midThreshold); % Red channel

% Display the color image for CT
figure;
imshow(coloredImage_ct1);
title('Intensity-Based Colored CT Image (CT Chest)');



% Color Image Generation for CT
% Convert to grayscale if necessary
if size(ct_image2, 3) == 3
    ct_image2 = rgb2gray(ct_image2);
end

% Normalize the image intensity to range [0, 1]
ct_image2 = mat2gray(ct_image2);

% Assign colors based on intensity levels
% Low intensity: Blue
% Medium intensity: Green
% High intensity: Red
coloredImage_ct2 = zeros([size(ct_image2), 3]); % Initialize a color image
coloredImage_ct2(:,:,3) = (ct_image2 <= lowThreshold); % Blue channel
coloredImage_ct2(:,:,2) = (ct_image2 > lowThreshold) & (ct_image2 <= midThreshold); % Green channel
coloredImage_ct2(:,:,1) = (ct_image2 > midThreshold); % Red channel

% Display the color image for CT
figure;
imshow(coloredImage_ct2);
title('Intensity-Based Colored CT Image (CT Head)');










% Read MRI and CT Wiener images
mri_image1 = imread('wiener_filtered_mri_of_brain2.jpg');
mri_image2 = imread('wiener_filtered_mri_of_brain1.jpg');
ct_image1 = imread('wiener_filtered_ctchest.jpg');
ct_image2 = imread('wiener_filtered_cthead.jpg');

% Store all images in a cell array
images = {mri_image1, mri_image2, ct_image1, ct_image2};

% Initialize cell array to store denoised images
denoised_images = cell(size(images));

% Perform wavelet denoising on each image
for i = 1:numel(images)
    % Convert to grayscale if necessary
    if size(images{i}, 3) == 3
        images{i} = rgb2gray(images{i});
    end

    % Perform Wavelet Denoising
    % 'sym4' is the chosen wavelet, and you can experiment with other wavelets
    % The level of decomposition is set to 1, but you can increase it as needed
    [denoisedImage,~] = wdenoise(double(images{i}), 1, 'Wavelet', 'sym4');

    % Convert back to uint8 if necessary
    denoised_images{i} = uint8(denoisedImage);
end

% Display the original and denoised images
figure;
for i = 1:numel(images)
    subplot(2, numel(images), i);
    imshow(images{i});
    title(['Original Image ', num2str(i)]);
   
    subplot(2, numel(images), i + numel(images));
    imshow(denoised_images{i});
    title(['Denoised Image ', num2str(i)]);
end

% Color Image Generation for MRI
% Convert to grayscale if necessary
if size(mri_image1, 3) == 3
    mri_image1 = rgb2gray(mri_image1);
end

% Normalize the image intensity to range [0, 1]
mri_image1 = mat2gray(mri_image1);

% Define intensity thresholds
lowThreshold = 0.33;
midThreshold = 0.66;

% Assign colors based on intensity levels
% Low intensity: Blue
% Medium intensity: Green
% High intensity: Red
coloredImage_mri1 = zeros([size(mri_image1), 3]); % Initialize a color image
coloredImage_mri1(:,:,3) = (mri_image1 <= lowThreshold); % Blue channel
coloredImage_mri1(:,:,2) = (mri_image1 > lowThreshold) & (mri_image1 <= midThreshold); % Green channel
coloredImage_mri1(:,:,1) = (mri_image1 > midThreshold); % Red channel

% Display the color image for MRI
figure;
imshow(coloredImage_mri1);
title('Intensity-Based Colored MRI Image (MRI of Brain 2)');

% Color Image Generation for MRI
% Convert to grayscale if necessary
if size(mri_image2, 3) == 3
    mri_image2 = rgb2gray(mri_image2);
end

% Normalize the image intensity to range [0, 1]
mri_image2 = mat2gray(mri_image2);

% Define intensity thresholds
lowThreshold = 0.33;
midThreshold = 0.66;

% Assign colors based on intensity levels
% Low intensity: Blue
% Medium intensity: Green
% High intensity: Red
coloredImage_mri2 = zeros([size(mri_image2), 3]); % Initialize a color image
coloredImage_mri2(:,:,3) = (mri_image2 <= lowThreshold); % Blue channel
coloredImage_mri2(:,:,2) = (mri_image2 > lowThreshold) & (mri_image2 <= midThreshold); % Green channel
coloredImage_mri2(:,:,1) = (mri_image2 > midThreshold); % Red channel

% Display the color image for MRI
figure;
imshow(coloredImage_mri2);
title('Intensity-Based Colored MRI Image (MRI of Brain 2)');

% Color Image Generation for CT
% Convert to grayscale if necessary
if size(ct_image1, 3) == 3
    ct_image1 = rgb2gray(ct_image1);
end

% Normalize the image intensity to range [0, 1]
ct_image1 = mat2gray(ct_image1);

% Assign colors based on intensity levels
% Low intensity: Blue
% Medium intensity: Green
% High intensity: Red
coloredImage_ct1 = zeros([size(ct_image1), 3]); % Initialize a color image
coloredImage_ct1(:,:,3) = (ct_image1 <= lowThreshold); % Blue channel
coloredImage_ct1(:,:,2) = (ct_image1 > lowThreshold) & (ct_image1 <= midThreshold); % Green channel
coloredImage_ct1(:,:,1) = (ct_image1 > midThreshold); % Red channel

% Display the color image for CT
figure;
imshow(coloredImage_ct1);
title('Intensity-Based Colored CT Image (CT Chest)');



% Color Image Generation for CT
% Convert to grayscale if necessary
if size(ct_image2, 3) == 3
    ct_image2 = rgb2gray(ct_image2);
end

% Normalize the image intensity to range [0, 1]
ct_image2 = mat2gray(ct_image2);

% Assign colors based on intensity levels
% Low intensity: Blue
% Medium intensity: Green
% High intensity: Red
coloredImage_ct2 = zeros([size(ct_image2), 3]); % Initialize a color image
coloredImage_ct2(:,:,3) = (ct_image2 <= lowThreshold); % Blue channel
coloredImage_ct2(:,:,2) = (ct_image2 > lowThreshold) & (ct_image2 <= midThreshold); % Green channel
coloredImage_ct2(:,:,1) = (ct_image2 > midThreshold); % Red channel

% Display the color image for CT
figure;
imshow(coloredImage_ct2);
title('Intensity-Based Colored CT Image (CT Head)');
```
