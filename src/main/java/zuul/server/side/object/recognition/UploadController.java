package zuul.server.side.object.recognition;

import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;


import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

@CrossOrigin(origins = "*", allowedHeaders = "*")
@RestController
@RequestMapping("/appliance")
public class UploadController {

	
    //Save the uploaded file to this folder
    private static String UPLOADED_FOLDER = "D://Upload//prova.jpg";

    @RequestMapping("/predict")
    public String singleFileUpload() {
    	
        String result = "";
        Path path = Paths.get(UPLOADED_FOLDER);
        //Recognize image
        result = Recognizer.recognize(path);
       
        return result;
    }

}