//...

//Parameters for loading the TensorFlow model in the app
String input, output;
AssetManager assetManager;
TensorFlowInferenceInterface tfHelper;
String modelPath, labelFile;
String[] outputs;

private static final long[] INPUT_SIZE = {1, 250,1};
private static final int OUTPUT_SIZE = 4;
Interpreter tflite;
//...
public Handler mainHandler;
private ThreadPoolExecutor mExecuter;
private final int MSG_DATA_CLASSIFICATION = 309;
private final int MSG_CANCEL = 310;
BlockingQueue<Runnable> workQueue;
private static int NUMBER_OF_CORES = Runtime.getRuntime().availableProcessors();
private static final int KEEP_ALIVE_TIME = 5;
//...

Public void onCreate(Bundle savedInstanceState){
	
//...
assetManager = getAssets();
input = "input";
output = "y_";
modelPath = "kfrozen_model_RNN_V2.pb";
labelFile = "labels.txt";
outputs = new String[]{output};
//...
loadProcessingSignal();
//...
workQueue = new LinkedBlockingQueue<Runnable>();
mExecuter = new ThreadPoolExecutor(NUMBER_OF_CORES,NUMBER_OF_CORES*2,KEEP_ALIVE_TIME,TimeUnit.SECONDS, workQueue);

//...
final Runnable classificationRunnable = new Runnable() {
            @Override
            public void run() {
                //---------------------------------------------
                String text = "";

               
       if (datareceivedfloat == null){
           Log.d("TF", "Data empty");

       }

        tfHelper.feed(input, datareceivedfloat, INPUT_SIZE);
        tfHelper.run(outputs);

        float[] result = new float[4];

        tfHelper.fetch(output, result);
                Log.d(TAG, "Classification Result" + Float.toString(result[0]) + " " +Float.toString(result[1])+ " " +Float.toString(result[2]) + " " +Float.toString(result[3]) );

                float val=0;
                int flag = 0;
                for (int k = 0; k<result.length;++k){
                    
                   if(val<result[k]){
                       val = result[k];
                       flag = k;
                   }
                }
                text = Integer.toString(flag);
                publicResult(text, ClassText);

        }

        };
//...
 mainHandler = new Handler() {
            @Override
            public void handleMessage(Message msg) {
                BluetoothGattCharacteristic characteristic;
                BluetoothGattDescriptor descriptor;
                switch (msg.what) {
                    case MSG_DATA_CLASSIFICATION:
                        datareceivedfloat = ((float[])(msg.obj));
                            if(datareceivedfloat == null){
                            Log.d(TAG, "Data null");
                        }
                        else{
                            Log.d(TAG, "Message recceived");
                            mExecuter.execute(classificationRunnable);       
                        }
                     break;
                    case MSG_CANCEL:
                        mExecuter.shutdown();
                     break;
                }
            }
};
//...

}

//...

//Method for loading the TensorFlow model in the app. It is done when the application is started.
private void loadProcessingSignal() {
       
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    
                    tfHelper = new TensorFlowInferenceInterface(assetManager,modelPath);

                } catch (final Exception e) {
                    
                    throw new RuntimeException("Error initializing classifiers!", e);
                }
            }
        }).start();
}

//...

//Method for showing result on the screen

public void updateResult(String result, TextView textView){
        textView.setText(result);
}
 

