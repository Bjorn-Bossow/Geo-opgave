function add_coastline(icoast)

if isfile(icoast)
    load(icoast);
    no_coastline = length(coastline);
else
    error(['File: ',icoast,' does not exist']);
end

for n = 1:no_coastline
    plot(coastline(n).X,coastline(n).Y,'color',[.2 .2 .2]);
end
