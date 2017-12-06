package darknet

// #cgo CFLAGS: -I ../include/
// #cgo LDFLAGS: -L .. -ldarknet
// #include <darknet.h>
// float get(float **a, int i, int j) { return a[i][j]; }
// box get_box(box *b, int i) { return b[i]; }
import "C"
import (
	"io/ioutil"
	"math"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"unsafe"

	"github.com/pkg/errors"
)

type Image struct {
	im C.struct___1
}

func (im *Image) Close() error {
	C.free_image(im.im)
	return nil
}

func LoadImage(path string) *Image {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	im := C.load_image_color(cpath, 0, 0)

	return &Image{
		im: im,
	}
}

type Network struct {
	Thresh, HierThresh, Nms float64

	Meta   NetworkMeta
	Labels []string

	n *C.struct_network
}

func (n *Network) Close() error {
	C.free_network(n.n)
	return nil
}

type NetworkMeta struct {
	Classes                           int
	Train, Valid, Names, Backup, Eval string
}

func (m *NetworkMeta) Decode(body []byte) error {
	lines := strings.Split(string(body), "\n")
	for _, l := range lines {
		l = strings.TrimSpace(l)
		if len(l) == 0 {
			continue
		}

		if l[:1] == "#" {
			continue
		}

		parts := strings.Split(l, "=")
		if len(parts) != 2 {
			return errors.Errorf("invalid line: %q", l)
		}

		k := strings.TrimSpace(parts[0])
		v := strings.TrimSpace(parts[1])
		if k == "names" {
			m.Names = v
		} else if k == "train" {
			m.Train = v
		} else if k == "valid" {
			m.Valid = v
		} else if k == "backup" {
			m.Backup = v
		} else if k == "eval" {
			m.Eval = v
		} else if k == "classes" {
			var err error
			m.Classes, err = strconv.Atoi(v)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func LoadNetwork(
	cfgPath, weightsPath, metaPath string, clear int,
) (*Network, error) {
	metaRaw, err := ioutil.ReadFile(resolvePath(metaPath))
	if err != nil {
		return nil, err
	}
	var meta NetworkMeta
	if err := meta.Decode(metaRaw); err != nil {
		return nil, err
	}

	labelsRaw, err := ioutil.ReadFile(resolvePath(meta.Names))
	if err != nil {
		return nil, err
	}
	labels := strings.Split(string(labelsRaw), "\n")[:meta.Classes]

	ccfg := C.CString(resolvePath(cfgPath))
	defer C.free(unsafe.Pointer(ccfg))
	cweights := C.CString(resolvePath(weightsPath))
	defer C.free(unsafe.Pointer(cweights))
	n := C.load_network(ccfg, cweights, C.int(clear))

	return &Network{
		Thresh:     DefaultThresh,
		HierThresh: DefaultHierThresh,
		Nms:        DefaultNms,
		n:          n,
		Meta:       meta,
		Labels:     labels,
	}, nil
}

var (
	DefaultThresh     = 0.01
	DefaultHierThresh = 0.5
	DefaultNms        = 0.45
)

type Detection struct {
	X, Y, W, H  float64
	Probability float64
	Label       string
}

// resolvePath resolves absolute paths relative to the root darknet directory.
func resolvePath(path string) string {
	if filepath.IsAbs(path) {
		return path
	}
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		return path
	}
	return filepath.Join(filepath.Dir(filename), "..", path)
}

func (n *Network) Detect(im *Image) ([]Detection, error) {
	boxes := C.make_boxes(n.n)
	probs := C.make_probs(n.n)
	num := C.num_boxes(n.n)
	defer C.free_ptrs((*unsafe.Pointer)(unsafe.Pointer(probs)), num)

	sized := C.letterbox_image(im.im, n.n.w, n.n.h)
	defer C.free_image(sized)

	w := float64(sized.w)
	h := float64(sized.h)

	C.network_detect(
		n.n, sized,
		C.float(n.Thresh), C.float(n.HierThresh), C.float(n.Nms),
		boxes,
		probs,
	)

	var ds []Detection
	for j := 0; j < int(num); j++ {
		box := C.get_box(boxes, C.int(j))
		for i, label := range n.Labels {
			p := float64(C.get(probs, C.int(j), C.int(i)))
			if p <= 0 {
				continue
			}

			if math.IsInf(p, 0) {
				continue
			}

			d := Detection{
				Label:       label,
				Probability: p,
				X:           float64(box.x),
				Y:           float64(box.y),
				W:           float64(box.w),
				H:           float64(box.h),
			}

			if d.W > w {
				d.W = w
			}
			if d.H > h {
				d.H = h
			}

			ds = append(ds)
		}
	}

	return ds, nil
}
