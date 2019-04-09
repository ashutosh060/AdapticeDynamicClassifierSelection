package DCS.models;

import java.io.File;
import java.io.FileWriter;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

public class CSVWriter {
    private static String produceCsvData(Object[] data) throws IllegalArgumentException, IllegalAccessException, InvocationTargetException {
        if (data.length == 0) {
            return "";
        }

        StringBuilder builder = new StringBuilder();

        for (Object d : data) {
            builder.append(d.toString());
            builder.append('\n');
        }
        builder.deleteCharAt(builder.length() - 1);
        return builder.toString();
    }

    public static boolean generateCSV(File csvFileName, Object[] data) {
        FileWriter fw = null;
        try {
            fw = new FileWriter(csvFileName);
            if (!csvFileName.exists())
                csvFileName.createNewFile();
            fw.write(produceCsvData(data));
            fw.flush();
        } catch (Exception e) {
            System.out.println("Error while generating csv from data. Error message : " + e.getMessage());
            e.printStackTrace();
            return false;
        } finally {
            if (fw != null) {
                try {
                    fw.close();
                } catch (Exception e) {
                }
                fw = null;
            }
        }
        return true;
    }
}