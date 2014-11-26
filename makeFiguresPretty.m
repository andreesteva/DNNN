figHandles = get(0,'Children');

for i =1:length(figHandles)
    figureHandle = figHandles(i);
    figureAxes = findall(figureHandle,'type','axes');
    % make all text in the figure to size 14 and bold
    set(findall(figureHandle,'type','text'),'fontSize',30,'fontWeight','bold') % Set text size
    set(get(gca,'xlabel'), 'FontSize', 20); % Set xlabel and ylabel fonts
    set(get(gca,'ylabel'), 'FontSize', 20); % Set xlabel and ylabel fonts
    set(figureAxes, 'fontsize', 20, 'linewidth', 2);
    set( findobj(gca, 'type', 'line'), 'LineWidth', 2);
   
end