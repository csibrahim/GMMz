function ok = heat(X,Y,C,step,res)

    rangeX = max(X)-min(X);
    rangeY = max(Y)-min(Y);
    
    stepX = rangeX*step;
    stepY = rangeY*step;

    r = round(Y/stepY);
    c = round(X/stepX);
    v = C;


    bs = full(sparse(r-min(r)+1,c-min(c)+1,1));
    if(~isempty(C))
        zs = full(sparse(r-min(r)+1,c-min(c)+1,v));
    
        zs = zs./bs;
        zs(isnan(zs))=0;
    else
        zs = log(bs+1);
    end

    [xs,ys] = meshgrid(1:size(zs,2),1:size(zs,1));
    [x,y] = meshgrid(1:res:size(zs,2),1:res:size(zs,1));

    z = interp2(xs,ys,zs,x,y,'makima');

    [y,x,z] = find(z);

    y = y+min(r)/res-1;
    x = x+min(c)/res-1;

    x = x*stepX*res;
    y = y*stepY*res;


    posX = [x-res*stepX/2 x-res*stepX/2 x+res*stepX/2 x+res*stepX/2];
    posY = [y+res*stepY/2 y-res*stepY/2 y-res*stepY/2 y+res*stepY/2];
    patch(posX',posY',z,'EdgeColor','None')
    colormap jet;
    
    axis tight;
end

