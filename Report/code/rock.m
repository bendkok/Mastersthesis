% My Rock
x = randn(1000, 1);
y = randn(1000, 1);
z = randn(1000, 1);

% Alpha Shape and plot
as = alphaShape(x, y, z, 4);
plot(as, 'FaceColor', [218 136 86]./255, 'EdgeAlpha', 0)
title('My Rock')
lighting gouraud
light('Position', [2 -4 2], 'Style', 'local')