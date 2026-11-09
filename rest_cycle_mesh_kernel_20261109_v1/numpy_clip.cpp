// Preserve NumPy matmul/add/sub exactly; compile only boolean/edge loops.
// Compile with -fno-fast-math -ffp-contract=off. No clearance approximation.
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <cmath>
#include <vector>

struct Plane { PyObject* normal; PyObject* offset; };
struct Planes { std::vector<Plane> rows; };
static void destroy(PyObject* capsule) {
    auto* p=static_cast<Planes*>(PyCapsule_GetPointer(capsule,"mesh.planes"));
    if(p) {for(auto& r:p->rows) {Py_DECREF(r.normal);Py_DECREF(r.offset);} delete p;}
}
static PyObject* prepare(PyObject*,PyObject* object) {
    PyArrayObject* eq=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(object,NPY_DOUBLE,NPY_ARRAY_CARRAY_RO));
    if(!eq) return nullptr;
    if(PyArray_NDIM(eq)!=2||PyArray_DIM(eq,1)!=4||PyArray_DIM(eq,0)<1||PyArray_DIM(eq,0)>12000) {
        Py_DECREF(eq);PyErr_SetString(PyExc_ValueError,"BOUNDED_PLANES_REQUIRED");return nullptr;
    }
    auto* p=new Planes();
    for(npy_intp i=0;i<PyArray_DIM(eq,0);i++) {
        PyObject* index=PyLong_FromSsize_t(i);
        PyObject* row=PyObject_GetItem(reinterpret_cast<PyObject*>(eq),index);Py_DECREF(index);
        PyObject* three=PyLong_FromLong(3);PyObject* slice=PySlice_New(nullptr,three,nullptr);
        PyObject* normal=row?PyObject_GetItem(row,slice):nullptr;
        PyObject* offset=row?PyObject_GetItem(row,three):nullptr;
        Py_XDECREF(row);Py_DECREF(slice);Py_DECREF(three);
        if(!normal||!offset) {
            Py_XDECREF(normal);Py_XDECREF(offset);Py_DECREF(eq);
            for(auto& r:p->rows) {Py_DECREF(r.normal);Py_DECREF(r.offset);} delete p;return nullptr;
        }
        p->rows.push_back({normal,offset});
    }
    Py_DECREF(eq);return PyCapsule_New(p,"mesh.planes",destroy);
}

static PyObject* clip(PyObject*,PyObject* args) {
    PyObject *input,*capsule;double expansion;
    if(!PyArg_ParseTuple(args,"OOd",&input,&capsule,&expansion)) return nullptr;
    auto* planes=static_cast<Planes*>(PyCapsule_GetPointer(capsule,"mesh.planes"));
    if(!planes) return nullptr;
    auto* polygon=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(input,NPY_DOUBLE,NPY_ARRAY_CARRAY_RO));
    if(!polygon) return nullptr;
    if(PyArray_NDIM(polygon)!=2||PyArray_DIM(polygon,0)!=3||PyArray_DIM(polygon,1)!=3) {
        Py_DECREF(polygon);PyErr_SetString(PyExc_ValueError,"TRIANGLE_REQUIRED");return nullptr;
    }
    PyObject* margin=PyFloat_FromDouble(expansion);
    for(const auto& plane:planes->rows) {
        npy_intp count=PyArray_DIM(polygon,0);
        if(!count) {Py_DECREF(margin);return reinterpret_cast<PyObject*>(polygon);}
        // These are the same three NumPy operations and scalar types as
        // polygon @ normal[:3] + normal[3] - expansion in the reference.
        PyObject* dot=PyNumber_MatrixMultiply(reinterpret_cast<PyObject*>(polygon),plane.normal);
        PyObject* shifted=dot?PyNumber_Add(dot,plane.offset):nullptr;Py_XDECREF(dot);
        PyObject* distance=shifted?PyNumber_Subtract(shifted,margin):nullptr;Py_XDECREF(shifted);
        if(!distance) {Py_DECREF(polygon);Py_DECREF(margin);return nullptr;}
        auto* ds=reinterpret_cast<PyArrayObject*>(distance);
        double* d=static_cast<double*>(PyArray_DATA(ds));
        bool all_in=true,all_out=true;bool finite=true;
        for(npy_intp i=0;i<count;i++) {all_in &= d[i]<=0;all_out &= d[i]>0;finite &= std::isfinite(d[i]);}
        if(!finite) {
            Py_DECREF(distance);Py_DECREF(polygon);Py_DECREF(margin);
            PyErr_SetString(PyExc_ValueError,"NONFINITE_CLIPPING_ARITHMETIC");return nullptr;
        }
        if(all_in) {Py_DECREF(distance);continue;}
        if(all_out) {
            Py_DECREF(distance);Py_DECREF(polygon);Py_DECREF(margin);
            npy_intp dims[2]={0,3};return PyArray_SimpleNew(2,dims,NPY_DOUBLE);
        }
        npy_intp next_count=0;
        for(npy_intp i=0;i<count;i++) {
            npy_intp j=(i+1)%count;next_count+=(d[i]<=0);next_count+=((d[i]<=0)!=(d[j]<=0));
        }
        if(next_count>static_cast<npy_intp>(planes->rows.size()+3)) {
            Py_DECREF(distance);Py_DECREF(polygon);Py_DECREF(margin);
            PyErr_SetString(PyExc_ValueError,"POLYGON_VERTEX_BOUND_EXCEEDED");return nullptr;
        }
        npy_intp dims[2]={next_count,3};
        auto* out=reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(2,dims,NPY_DOUBLE));
        if(!out) {Py_DECREF(distance);Py_DECREF(polygon);Py_DECREF(margin);return nullptr;}
        auto* p=static_cast<double*>(PyArray_DATA(polygon));auto* dst=static_cast<double*>(PyArray_DATA(out));
        npy_intp n=0;
        for(npy_intp i=0;i<count;i++) {
            npy_intp j=(i+1)%count;
            if(d[i]<=0) {for(int k=0;k<3;k++) dst[3*n+k]=p[3*i+k];n++;}
            if((d[i]<=0)!=(d[j]<=0)) {
                const double denominator=d[i]-d[j];const double ratio=d[i]/denominator;
                for(int k=0;k<3;k++) {
                    const double delta=p[3*j+k]-p[3*i+k];const double product=ratio*delta;
                    dst[3*n+k]=p[3*i+k]+product;
                }
                n++;
            }
        }
        Py_DECREF(distance);Py_DECREF(polygon);polygon=out;
    }
    Py_DECREF(margin);return reinterpret_cast<PyObject*>(polygon);
}
static PyMethodDef methods[]={{"prepare",prepare,METH_O,nullptr},{"clip",clip,METH_VARARGS,nullptr},{nullptr,nullptr,0,nullptr}};
static PyModuleDef module={PyModuleDef_HEAD_INIT,"numpy_clip",nullptr,-1,methods};
PyMODINIT_FUNC PyInit_numpy_clip() {import_array();return PyModule_Create(&module);}
