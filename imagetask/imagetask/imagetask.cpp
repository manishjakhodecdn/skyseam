#include <half.h>
#include <stdio.h>
#include <ImfInputFile.h>
#include <ImfAttribute.h>
#include <ImfBoxAttribute.h>
#include <ImfChannelListAttribute.h>
#include <ImfChromaticitiesAttribute.h>
#include <ImfCompressionAttribute.h>
#include <ImfDoubleAttribute.h>
#include <ImfEnvmapAttribute.h>
#include <ImfFloatAttribute.h>
#include <ImfIntAttribute.h>
#include <ImfKeyCodeAttribute.h>
#include <ImfLineOrderAttribute.h>
#include <ImfMatrixAttribute.h>
#include <ImfPreviewImageAttribute.h>
#include <ImfRationalAttribute.h>
#include <ImfStringAttribute.h>
#include <ImfStringVectorAttribute.h>
#include <ImfTileDescriptionAttribute.h>
#include <ImfTimeCodeAttribute.h>
#include <ImfVecAttribute.h>
#include <ImfVersion.h>
#include <ImathBox.h>
#include <string>

#include <ImfRgbaFile.h>
#include <ImfPreviewImage.h>

#include <imagetask.h>

using namespace Imf;

int read_exr_info( char **error_p, 
                   char *path,
                   bool read_buffer,
                   Missing_Type_Function_p missing_type_function_p,
                   Alloc_Image_Buffer_Function_p alloc_image_buffer_function_p,
                   Named_Int_Function_p named_int_function_p,
                   Named_Float_Function_p named_float_function_p,
                   Named_Str_Function_p named_str_function_p,
                   Named_M44f_Function_p named_m44f_function_p,
                   Named_Box2i_Function_p named_box2i_function_p,
                   Named_Channel_Function_p named_Channel_function_p){
    try{
        RgbaInputFile exrfile(path);
        
        if( read_buffer ){
            Imath::Box2i dw = exrfile.dataWindow();
            int fullwidth = dw.max.x - dw.min.x + 1;
            int fullheight = dw.max.y - dw.min.y + 1;        
            
            int width, height;
            void *dst = alloc_image_buffer_function_p( fullwidth, fullheight, width, height);

            Array2D <PreviewRgba> previewPixels;
            make_preview(exrfile,
                         1.0,
                         width,
                         height,
                         previewPixels);
            
            void *src = (void *)&(previewPixels[0][0]);
            
            memcpy( dst, src, width*height*4);
        }
            
        const Header &h = exrfile.header();
        int rsize = 0;
        for (Header::ConstIterator i = h.begin(); i != h.end(); ++i, ++rsize){
            const Attribute *a = &i.attribute();
            
            if (const IntAttribute *ta = dynamic_cast <const IntAttribute *> (a)){
                named_int_function_p( (char *)i.name(), ta->value());
            }
            else if (const FloatAttribute *ta = dynamic_cast <const FloatAttribute *> (a)){
                named_float_function_p( (char *)i.name(), ta->value());
            }
            else if (const M44fAttribute *ta = dynamic_cast <const M44fAttribute *> (a)){
                named_m44f_function_p( (char *)i.name(), 
                        ta->value()[0][0], ta->value()[0][1], ta->value()[0][2], ta->value()[0][3], 
                        ta->value()[1][0], ta->value()[1][1], ta->value()[1][2], ta->value()[1][3], 
                        ta->value()[2][0], ta->value()[2][1], ta->value()[2][2], ta->value()[2][3], 
                        ta->value()[3][0], ta->value()[3][1], ta->value()[3][2], ta->value()[3][3]);
            }
            else if (const StringAttribute *ta = dynamic_cast <const StringAttribute *> (a)){
                named_str_function_p( (char *)i.name(), (char *) ta->value().c_str());
            }
            else if (const Box2iAttribute *ta = dynamic_cast <const Box2iAttribute *> (a)){
                named_box2i_function_p( (char *)i.name(), 
                                       ta->value().min[0],
                                       ta->value().min[1],
                                       ta->value().max[0],
                                       ta->value().max[1]);
            }
            else if (const ChannelListAttribute *ta = dynamic_cast <const ChannelListAttribute *> (a)){
                const ChannelList &cl = ta->value();
                for (ChannelList::ConstIterator i = cl.begin(); i != cl.end(); ++i){
                    std::string value;
                    switch (i.channel().type){
                        case UINT:
                            value = "unit"; break;
                        case HALF:
                            value = "half"; break;
                        case FLOAT:
                            value = "float"; break;
                        default:
                            value = "unknown"; break;
                    }
                    named_Channel_function_p( (char *)i.name(), (char *) value.c_str() );
                }
            }
            else if (const CompressionAttribute *ta =
                     dynamic_cast <const CompressionAttribute *> (a))
            {
                std::string value;
                switch (ta->value()){
                    case NO_COMPRESSION:
                        value = "none"; break;
                    case RLE_COMPRESSION:
                        value = "run-length encoding"; break;
                    case ZIPS_COMPRESSION:
                        value = "zip, individual scanlines"; break;
                    case ZIP_COMPRESSION:
                        value =  "zip, multi-scanline blocks"; break;
                    case PIZ_COMPRESSION:
                        value =  "piz"; break;
                    case PXR24_COMPRESSION:
                        value =  "pxr24"; break;
                    case B44_COMPRESSION:
                        value =  "b44"; break;
                    case B44A_COMPRESSION:
                        value =  "b44a"; break;
                    default:
                        value =  "unknown"; break;
                }
                named_str_function_p( (char *)i.name(), (char *) value.c_str());
            }
            else{
                missing_type_function_p( (char *)i.name(), (char *) a->typeName() );
            }
        }
    }
    catch (const std::exception &e){
        *error_p = const_cast< char *>( e.what());
        return 666;
    }
    return 0;
}
