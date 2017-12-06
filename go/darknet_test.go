package darknet

import "testing"

func TestDetect(t *testing.T) {
	n, err := LoadNetwork("cfg/yolo.cfg", "yolo.weights", "cfg/coco.data", 0)
	if err != nil {
		t.Fatal(err)
	}
	defer n.Close()
	im := LoadImage("../data/dog.jpg")
	defer im.Close()
	d, err := n.Detect(im)
	if err != nil {
		t.Fatal(err)
	}
	if len(d) == 0 {
		t.Fatalf("detections is empty: %+v", d)
	}
	t.Logf("Detections: %+v", d)
}
