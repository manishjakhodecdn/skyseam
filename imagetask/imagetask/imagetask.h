#include <makepreview.h>

typedef int (*Missing_Type_Function_p)(char *, char *);
typedef void *(*Alloc_Image_Buffer_Function_p)(int, int, int &, int &);
typedef int (*Named_Int_Function_p)(char *,int);
typedef int (*Named_Float_Function_p)(char *, float);
typedef int (*Named_Str_Function_p)(char *, char *);
typedef int (*Named_Box2i_Function_p)(char *, int, int, int, int);
typedef int (*Named_Channel_Function_p)(char *, char *);
typedef int (*Named_M44f_Function_p)(char *, float,float,float,float,
                                             float,float,float,float,
                                             float,float,float,float,
                                             float,float,float,float );

extern "C" int read_exr_info( char **error_p, 
                              char *path, 
                              bool read_buffer,
                              Missing_Type_Function_p missing_type_function_p,
                              Alloc_Image_Buffer_Function_p,
                              Named_Int_Function_p named_int_function_p,
                              Named_Float_Function_p named_float_function_p,
                              Named_Str_Function_p named_str_function_p,
                              Named_M44f_Function_p named_m44f_function_p,
                              Named_Box2i_Function_p named_box2i_function_p,
                              Named_Channel_Function_p named_Channel_function_p);