function obj = getShape(type)
switch lower(type)
    case {'linhex','hexa','hex8','8'}
        obj = ShapeFunctions.hex8.Hex8();
    case {'lintet','tet4','4'}
        obj = ShapeFunctions.tet4.Tet4();
    case {'quadtet','tet10','10'}
        obj = ShapeFunctions.tet10.Tet10();
    otherwise
        error('Unknown shape type "%s".', type);
end
end