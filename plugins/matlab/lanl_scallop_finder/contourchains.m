function edgelist = contourchains(I) %#codegen

% global ROWS;
% global COLS;
% global roff;
% global coff;
%This traversal for open contours
roff = [1  0 -1  0  1  1 -1 -1];
coff = [0  1  0 -1  1 -1 -1  1];
%This traversal for closed contours
% roff = [-1  0  1  1  1  0 -1 -1];
% coff = [ 1  1  1  0 -1 -1 -1  0];

[ROWS, COLS] = size(I);
edgeNo = 1;
n = sum(I(:));
edgelist = zeros(n,4);
offset = 0;
thereIsAPoint = 0;
nextrow = 1;
nextcol = 1;
for row = 1:ROWS
    for col = 1:COLS
        if I(row,col)
            edgepoints = zeros(2*(ROWS+COLS),2);
            count = 1;
            edgepoints(count,:) = [row col];
            I(row,col) = 0;
            r = row+roff;
            c = col+coff;
            flag = 0;
            for i = 1:8
                if r(i) >= 1 && r(i) <= ROWS && c(i) >= 1 && c(i) <= COLS
                    if I(r(i),c(i)) == 1
                        flag = 1;
                        nextrow = r(i);
                        nextcol = c(i);
                        thereIsAPoint = 1;
                        break;
                    end
                end
            end
            if flag == 0
                nextrow = 0;
                nextcol = 0;
                thereIsAPoint = 0;
            end

            while thereIsAPoint
                count = count+1;
                edgepoints(count,:) = [nextrow nextcol];
                I(nextrow,nextcol) = 0;
                r = nextrow+roff;
                c = nextcol+coff;
                flag = 0;
                for i = 1:8
                    if r(i) >= 1 && r(i) <= ROWS && c(i) >= 1 && c(i) <= COLS
                        if I(r(i),c(i)) == 1
                            flag = 1;
                            nextrow = r(i);
                            nextcol = c(i);
                            thereIsAPoint = 1;
                            break;
                        end
                    end
                end
                if flag ==0
                    nextrow = 0;
                    nextcol = 0;
                    thereIsAPoint = 0;
                end
            end
            edgepoints(1:count,:) = flipud(edgepoints(1:count,:));
            r = row+roff;
            c = col+coff;
            flag = 0;
            for i = 1:8
                if r(i) >= 1 && r(i) <= ROWS && c(i) >= 1 && c(i) <= COLS
                    if I(r(i),c(i)) == 1
                        flag = 1;
                        nextrow = r(i);
                        nextcol = c(i);
                        thereIsAPoint = 1;
                        break;
                    end
                end
            end
            if flag == 0
                nextrow = 0;
                nextcol = 0;
                thereIsAPoint = 0;
            end

            while thereIsAPoint
                count = count+1;
                edgepoints(count,:) = [nextrow nextcol];
                I(nextrow,nextcol) = 0;
                r = nextrow+roff;
                c = nextcol+coff;
                flag = 0;
                for i = 1:8
                    if r(i) >= 1 && r(i) <= ROWS && c(i) >= 1 && c(i) <= COLS
                        if I(r(i),c(i)) == 1
                            flag = 1;
                            nextrow = r(i);
                            nextcol = c(i);
                            thereIsAPoint = 1;
                            break;
                        end
                    end
                end
                if flag == 0
                    nextrow = 0;
                    nextcol = 0;
                    thereIsAPoint = 0;
                end
            end
            edgepoints = edgepoints(1:count,:);
            if ~isempty(edgepoints)
                s = size(edgepoints,1);
                edgelist(offset+1:offset+s,:) = [edgepoints repmat([edgeNo s],s,1)];
                edgeNo = edgeNo + 1;
                offset = offset+s;
            end
        end
    end
end
