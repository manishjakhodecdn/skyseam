import exrinfo
from xml.etree import ElementTree

def convertElement(o):
    d = dict(o.attrib)
    if o.text and o.text.strip():
        d[''] = o.text.strip()
    for child in o:
        d.setdefault(child.tag, []).append(child)
    return d

simple_type_names = ['string', 'int', 'double', 'float']
def simple_value_from_type_string( kind, value ):
    if kind == 'string':
        return str(value)
    elif kind == 'int':
        return int(value)
    elif kind == 'double':
        return float(value)
    elif kind == 'float':
        return float(value)
    raise Warning('unkown type')

def collapseScopes( d ):
    if isinstance( d, dict):
        if len(d.keys() ) == 1 and '' in d:
            return d['']
                                                                      
        elif 'hash_map' in d:
            nvalue = d['hash_map'][0]['entry']
            name = d.get('id')
            
            result = {}
            for child in nvalue:
                for key, value in child.iteritems():
                    result[key] = value
            d = {}
            if name is not None:
                d[name] = result
            else:
                d = result

        elif len(d.keys() ) == 2 and 'id' in d:
            keys = d.keys()
            keys.remove('id')
            name = d['id']
            kind = keys[-1]
            value = d[kind]
            
            if kind in simple_type_names:
                d = {}
                d[name] = simple_value_from_type_string(kind, value[0])
                
            elif kind == 'vector':
                vector = value[0]
                vector.pop( 'size_t' )
                vector.pop( 'subtype' )
                item_kind, items = vector.items()[0]
                vector_result = []
                for item_value in items:
                    vector_result.append( simple_value_from_type_string(item_kind, item_value))
                d = {}
                d[name] = vector_result
            
            elif kind == 'GfVec2d':
                item_kind, item_struct = value[0].items()[0]
                items = item_struct[0][''].strip().split()
                vector_result = []
                for item_value in items:
                    vector_result.append( simple_value_from_type_string(item_kind, item_value))
                d = {}
                d[name] = vector_result

    return d

def convertExtendedInfo( element ):
    if isinstance(element, ElementTree.Element):
        d = convertElement( element )
        if isinstance( d, dict):
            for key, value in d.iteritems():
                itemiter = None
                if isinstance(value, list):
                    itemiter = enumerate(value)
                elif isinstance(value, dict):
                    itemiter = value.iteritems
                
                if itemiter:
                    for child_key, child_value in itemiter:
                        value[child_key] = convertExtendedInfo( child_value )
    else:
        d = element
        
    d = collapseScopes( d )
    return d


class ExrInfoExt( exrinfo.ExrInfo ):
    def __init__( self, path, read_buffer=True, aspect=None, max_width=256, max_height=256 ):
        super( ExrInfoExt, self).__init__( path, read_buffer=read_buffer )

        ee_str  = self.attributes.get( 'extendedInfo', None )

        if ee_str:
            ee_xml  = ElementTree.fromstring(ee_str)
            ee_dict = convertExtendedInfo( ee_xml )
            self.attributes[ 'extendedInfo' ] = ee_dict
