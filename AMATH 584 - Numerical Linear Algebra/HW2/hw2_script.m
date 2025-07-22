% Load the image.
I = imread('cameraman.tif');

% Convert image to double precision.
A = im2double(I);

% Perform SVD for the matrix A such that A = U*S*V'.
[U,S,V] = svd(A);

% Show the full rank image.
figure;
imshow(A); 
title('Full Rank Image');

% Truncate the SVD.
% Rank 1:

% Copy S and only keep the first singular value.
S_1 = S;
S_1(:, 2:end) = zeros(size(A, 1), size(A, 2)-1);

% Use SVD formula to get Rank 1 approximation to A.
A_1 = U*S_1*V';
figure;
imshow(A_1)
title("Rank 1 Image")

% The same procedure that was used above can be used again with 1 replaced with k if we want a rank-k image. 

% Rank 10:
S_10 = S;
S_10(:, 11:end) = zeros(size(A, 1), size(A, 2)-10);
A_10 = U*S_10*V';
figure;
imshow(A_10)
title("Rank 10 Image");

% Rank 50:
S_50 = S;
S_50(:, 51:end) = zeros(size(A, 1), size(A, 2)-50);
A_50 = U*S_50*V';
figure;
imshow(A_50)
title("Rank 50 Image")

% Rank 150:
figure;
S_150 = S;
S_150(:, 151:end) = zeros(size(A, 1), size(A, 2)-150);
A_150 = U*S_150*V';
imshow(A_150)
title("Rank 150 Image");



