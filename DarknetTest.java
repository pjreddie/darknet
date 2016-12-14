class DarknetTest {
    public static void main(String args[]) {
        
        Runtime rt = Runtime.getRuntime();
        Process pr = rt.exec("./darknet detector test cfg/coco.data cfg/yolo.cfg yolo.weights");
    }
}
