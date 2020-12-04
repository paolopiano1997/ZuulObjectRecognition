package zuul.server.side.object.recognition;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class Recognizer {

	//private static String base_dir = "D:\\Documenti\\Corsi di Studio Magistrale\\Corsi di studio Secondo Anno\\Primo Semestre\\Sistemi Distribuiti\\Progetto\\ObjectRecognition\\inception_dec_2015";
	
	//private static String base_dir = "D:\\Documenti\\Paolo\\Sistemi Distribuiti\\Progetto\\ObjectRecognition\\inception_dec_2015"; //PC fisso

	private static String base_dir = "/home/debian/inception_dec_2015";
	
	private static byte[] graphDef = readAllBytesOrExit(Paths.get(base_dir, "tensorflow_inception_graph.pb"));

    private static List<String> labels  = readAllLinesOrExit(Paths.get(base_dir, "imagenet_comp_graph_label_strings.txt"));
    
    
    public static String recognize(Path path ) {
    	String result = "";
	    byte[] imageBytes = readAllBytesOrExit(path);
	    try (Tensor image = Tensor.create(imageBytes)) {
	        float[] labelProbabilities = executeInceptionGraph(graphDef, image);
	        int bestLabelIdx = maxIndex(labelProbabilities);
	        result = String.format(
	                "BEST MATCH: %s (%.2f%% likely)",
	                labels.get(bestLabelIdx), labelProbabilities[bestLabelIdx] * 100f);
	        System.out.println(result);
	    }
	    return result;
    }
    
    private static int maxIndex(float[] probabilities) {
        int best = 0;
        for (int i = 1; i < probabilities.length; ++i) {
            if (probabilities[i] > probabilities[best]) {
                best = i;
            }
        }
        return best;
    }
    
    private static byte[] readAllBytesOrExit(Path path) {
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(1);
        }
        return null;
    }
    
    private static List<String> readAllLinesOrExit(Path path) {
        try {
            return Files.readAllLines(path, Charset.forName("UTF-8"));
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(0);
        }
        return null;
    }
    
    private static float[] executeInceptionGraph(byte[] graphDef, Tensor image) {
        try (Graph g = new Graph()) {
            g.importGraphDef(graphDef);
            try (Session s = new Session(g);
                    Tensor result = s.runner().feed("DecodeJpeg/contents", image).fetch("softmax").run().get(0)) {
                final long[] rshape = result.shape();
                if (result.numDimensions() != 2 || rshape[0] != 1) {
                    throw new RuntimeException(
                            String.format(
                                    "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                                    Arrays.toString(rshape)));
                }
                int nlabels = (int) rshape[1];
                return result.copyTo(new float[1][nlabels])[0]; //Da controllare
            }
        }
    }
    
}
