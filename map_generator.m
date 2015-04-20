map = imread('map.bmp');
r = size(map, 1);
c = size(map, 2);
book = zeros(r*c, 3);
for r1=1:r
    for c1=1:c
        book((r1-1)*c+c1, 1) = r1 - 1;
        book((r1-1)*c+c1, 2) = c1 - 1;
        book((r1-1)*c+c1, 3) = map(r1, c1);
    end
end
csvwrite('map.csv', book);