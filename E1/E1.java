public class Sensor implements Handler.Callback  {

    public BluetoothGatt mBluetoothGatt;
    public BluetoothDevice mDevice;
    public boolean isConnectDevice = false;
    public boolean mAquisitionStarted = false;
    public Context mContext;
    public String TAG = "Sensor.class";
    private Handler bleHandler;
    private MyBleCallback myBleCallback = new MyBleCallback();
    public ConcurrentLinkedQueue<BioPotDataBean> cQueue =  new ConcurrentLinkedQueue<>();
    public CopyOnWriteArrayList<float[]> DataForClassification = new CopyOnWriteArrayList<>();
    Handler mainHandler;
    //--------------------------------UUID-----------------------------------------------------------
    private final UUID BIOPOTPROFILE_SERV_UUID = UUID.fromString("0000FFF0-0000-1000-8000-00805F9B34FB");
    private final UUID BIOPOTPROFILE_CHAR1_UUID = UUID.fromString("0000FFF1-0000-1000-8000-00805F9B34FB");
    private final UUID BIOPOTPROFILE_CHAR2_UUID = UUID.fromString("0000FFF2-0000-1000-8000-00805F9B34FB");
    private final UUID BIOPOTPROFILE_CHAR3_UUID = UUID.fromString("0000FFF3-0000-1000-8000-00805F9B34FB");
    private final UUID BIOPOTPROFILE_CHAR4_UUID = UUID.fromString("0000FFF4-0000-1000-8000-00805F9B34FB");
    private final UUID BIOPOTPROFILE_CHAR4_CONFIG_UUID = UUID.fromString("00002902-0000-1000-8000-00805F9B34FB");
    private final UUID BIOPOTPROFILE_CHAR5_UUID = UUID.fromString("0000FFF5-0000-1000-8000-00805F9B34FB");
    //----------------------MSG CODE------------------------
    private final int MSG_DISCONNECT = 302;
    private final int MSG_CONNECT = 303;
    private final int MSG_DATA = 305;
    private final int MSG_SERVICE_DISCOVERED= 306;
    private final int MSG_DISCONNECTED = 307;
    private final int MSG_CONNECTED = 308;
    private final int MSG_DATA_CLASSIFICATION = 309;
  

    public Sensor(BluetoothDevice device, Context mAppContext, Handler mainHandler){
        mDevice = device;
        mContext = mAppContext;
        HandlerThread handlerThread = new HandlerThread("BleThread");
        handlerThread.start();
        bleHandler = new Handler(handlerThread.getLooper(), this);
        mDevice = device;
        mBluetoothGatt = mDevice.connectGatt(context, false, myBleCallback);
        this.mainHandler = mainHandler;
        Log.d(TAG,"Sensor element created");
    }

    

  Runnable sendingData = new Runnable() {
        @Override
        public void run() {
            if (mAquisitionStarted){
                Message data = new Message();
                data.what = MSG_DATA_CLASSIFICATION;
                mainHandler.sendMessageAtFrontOfQueue(Message.obtain(null, MSG_DATA_CLASSIFICATION, cQueue));
                erasecQueue();
                bleHandler.postDelayed(this, 100);
                Log.d(TAG, "Message sent");
            }
            else{
                Log.d(TAG, "Acquisition not started");
            }

        }};
    public CopyOnWriteArrayList<float[]> getData(){
        return DataForClassification;
    }
    public void Connect(){
        isConnectDevice = true;
        mBluetoothGatt = mDevice.connectGatt(mContext,false,myBleCallback);
    }

    public void Disconnect(){
        mBluetoothGatt.close();
    }
    public void acquisition(boolean startAcq){
        final byte[] ss = new byte[] { (byte)0x1};
        if (!startAcq) ss[0] = (byte) 0;
        if (!startAcq)  mAquisitionStarted = false;
        if (startAcq) {
            enableNotification();
            SystemClock.sleep(400);
        }
        if (( mBluetoothGatt != null) && (mDevice != null)) {
            BluetoothGattCharacteristic characteristic;
            BluetoothGattService service0;
            Log.i(TAG, "start acquistion: setnotification and send '1' ");
            service0 =  mBluetoothGatt.getService(BIOPOTPROFILE_SERV_UUID);
            if (service0 != null) {
                characteristic = service0.getCharacteristic(BIOPOTPROFILE_CHAR2_UUID);
                characteristic.setValue(ss);
                boolean retflag =  mBluetoothGatt.writeCharacteristic(characteristic);
                if (startAcq) mAquisitionStarted = true;
                bleHandler.post(sendingData);
            }
        }
    }
    public void startAcquisition(){
        acquisition(true);
    }
    public void stopAcquisition(){
        acquisition(false);
    }
    public void enableNotification()
    {
        if (( mBluetoothGatt !=null) && (mDevice !=null)) {
            BluetoothGattCharacteristic characteristic;
            BluetoothGattService service0;
            Log.i(TAG, "Set notify");
            service0 =  mBluetoothGatt.getService(BIOPOTPROFILE_SERV_UUID);
            if (service0 != null) {
                characteristic = service0.getCharacteristic(BIOPOTPROFILE_CHAR4_UUID);
                //Enable local notifications
                mBluetoothGatt.setCharacteristicNotification(characteristic, true);
                //Enabled remote notifications
                BluetoothGattDescriptor desc = characteristic.getDescriptor(BIOPOTPROFILE_CHAR4_CONFIG_UUID);
                desc.setValue(BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE);
                mBluetoothGatt.writeDescriptor(desc);
            }
        }
    }
    public interface BleCallback {
        /**
         * Signals that the BLE device is ready for communication.
         */
        @MainThread
        void onDeviceReady();
        /**
         * Signals that a connection to the device was lost.
         */
        @MainThread
        void onDeviceDisconnected();
    }
	
    @Override
    public boolean handleMessage(Message msg) {
        switch (msg.what){

            case MSG_CONNECT:
                Log.d(TAG, "Connecting to GATT server.");
                break;

            case MSG_CONNECTED:
                Log.d(TAG, "Connected to GATT server.");
                ((BluetoothGatt) msg.obj).discoverServices();
                break;

            case MSG_DISCONNECT:
                Log.d(TAG, "Disconnected from GATT server.");
                ((BluetoothGatt) msg.obj).disconnect();
                break;
            case MSG_DISCONNECTED:
                Log.d(TAG, "Disconnecting from GATT server.");
                ((BluetoothGatt) msg.obj).close();;
                break;

            case MSG_DATA:
                BluetoothGattCharacteristic characteristic = ((BluetoothGattCharacteristic) msg.obj);
                BioPotDataBean bd = new BioPotDataBean();
                byte[] pValue1 = characteristic.getValue();
                bd.setData(characteristic.getValue(), (short) pValue1.length);
                cQueue.add(bd);
                break;

            case MSG_SERVICE_DISCOVERED:

                break;
        }
        return true;
    }

    public ConcurrentLinkedQueue<BioPotDataBean> getcQueue(){

        return cQueue;
    }

    public void erasecQueue(){
        cQueue.clear();
    }
	
    public void transformData(BluetoothGattCharacteristic characteristic){
        BioPotDataBean bd = new BioPotDataBean();
        byte[] pValue1 = characteristic.getValue();
        bd.setData(characteristic.getValue(), (short) pValue1.length);
        short[] Value = bd.getData();
        float[] valArr = new float[8];

        for (int j=0; j< pValue1.length; j=j+8) {

            valArr[0] = (float) (((double) (pValue1[j])) * 0.000195);// + 8 * 4;  //+2 removing type, timestamp and len ...

            valArr[1]= (float) (((double) (pValue1[1+j])) * 0.000195);//+ 8 * 3;  //+2 removing type, timestamp and len ...

            valArr[2] = (float) (((double) (pValue1[2+j])) * 0.000195);//+ 8 * 2;  //+2 removing type, timestamp and len ...

            valArr[3] = (float) (((double) (pValue1[3+j])) * 0.000195);//+ 8 * 1;  //+2 removing type, timestamp and len ...

            valArr[4] = (float) (((double) (pValue1[4+j])) * 0.000195);//+ 8 * 0;  //+2 removing type, timestamp and len ...

            valArr[5] = (float) (((double) (pValue1[5+j])) * 0.000195);//- 8 * 1;  //+2 removing type, timestamp and len ...

            valArr[6] = (float) (((double) (pValue1[6+j])) * 0.000195);//- 8 * 2;  //+2 removing type, timestamp and len ...

            valArr[7] = (float) (((double) (pValue1[7+j])) * 0.000195);//- 8 * 2;  //+2 removing type, timestamp and len ...

            valArr[0] = (float) (((double) (pValue1[7+j])) * 0.000195);//- 8 * 3;  //+2 removing type, timestamp and len ...

            DataForClassification.add(valArr);
        }
    }

    private class MyBleCallback extends BluetoothGattCallback {

        @Override
        public void onConnectionStateChange(BluetoothGatt gatt, int status, int newState) {
            super.onConnectionStateChange(gatt, status, newState);
            if (status == BluetoothGatt.GATT_SUCCESS && newState == BluetoothProfile.STATE_CONNECTED) {
                isConnectDevice = true;
                bleHandler.obtainMessage(MSG_CONNECTED, gatt).sendToTarget();

                } else if (status == BluetoothGatt.GATT_SUCCESS && newState == BluetoothProfile.STATE_DISCONNECTED) {
                
                bleHandler.obtainMessage(MSG_DISCONNECTED, gatt).sendToTarget();
            } else if (status != BluetoothGatt.GATT_SUCCESS) {
                isConnectDevice = false;
                bleHandler.obtainMessage(MSG_DISCONNECT, gatt).sendToTarget();
            }
        }

        @Override
        public void onServicesDiscovered(BluetoothGatt gatt, int status) {
            super.onServicesDiscovered(gatt, status);
            Log.i(TAG, "Services Discovered: " + status);
        }

        @Override
        public void onCharacteristicRead(BluetoothGatt gatt, BluetoothGattCharacteristic characteristic, int status) {
            super.onCharacteristicRead(gatt, characteristic, status);

            Log.i(TAG, "onCharacteristicRead " + Arrays.toString(characteristic.getValue()));
            if (BIOPOTPROFILE_CHAR1_UUID.equals(characteristic.getUuid())) {
                Log.i(TAG, "BIOPOTPROFILE_CHAR1_UUID");
                Log.i(TAG, "getProperties " + characteristic.getProperties());

            } else if (BIOPOTPROFILE_CHAR2_UUID.equals(characteristic.getUuid())) {
                Log.i(TAG, "BIOPOTPROFILE_CHAR2_UUID");
                Log.i(TAG, "getProperties " + characteristic.getProperties());

            } else if (BIOPOTPROFILE_CHAR3_UUID.equals(characteristic.getUuid())) {
                Log.i(TAG, "BIOPOTPROFILE_CHAR3_UUID");
                Log.d(TAG, "getProperties " + characteristic.getProperties());

            } else if (BIOPOTPROFILE_CHAR5_UUID.equals(characteristic.getUuid())) {
                Log.i(TAG, "BIOPOTPROFILE_CHAR5_UUID");
                Log.i(TAG, "getProperties " + characteristic.getProperties());

            }
            else if (BIOPOTPROFILE_CHAR2_UUID.equals(characteristic.getUuid())) {
               
            } 
        }

        @Override
        public void onCharacteristicWrite(BluetoothGatt gatt, BluetoothGattCharacteristic characteristic, int status) {
            super.onCharacteristicWrite(gatt, characteristic, status);
            Log.i(TAG, "onCharacteristicWrite ");
        }
		
        @Override
        public void onCharacteristicChanged(BluetoothGatt gatt, BluetoothGattCharacteristic characteristic) {
            super.onCharacteristicChanged(gatt, characteristic);
            if (BIOPOTPROFILE_CHAR4_UUID.equals(characteristic.getUuid())) {
                bleHandler.obtainMessage(MSG_DATA, characteristic).sendToTarget();
            }
        }

        @Override
        public void onDescriptorRead(BluetoothGatt gatt, BluetoothGattDescriptor descriptor, int status) {
            super.onDescriptorRead(gatt, descriptor, status);
            Log.i(TAG, "onDescriptorRead " + Arrays.toString(descriptor.getValue()));
        }

        @Override
        public void onDescriptorWrite(BluetoothGatt gatt, BluetoothGattDescriptor descriptor, int status) {
            super.onDescriptorWrite(gatt, descriptor, status);

            Log.i(TAG, "onDescriptorWrite " + Arrays.toString(descriptor.getValue()));
            gatt.requestMtu(251);
        }

        @Override
        public void onReadRemoteRssi(BluetoothGatt gatt, int rssi, int status) {
            super.onReadRemoteRssi(gatt, rssi, status);
        }

        @Override
        public void onMtuChanged(BluetoothGatt gatt, int mtu, int status) {
            super.onMtuChanged(gatt, mtu, status);
        }
        private String connectionState(int status) {
            switch (status) {
                case BluetoothProfile.STATE_CONNECTED:
                    return "Connected";
                case BluetoothProfile.STATE_DISCONNECTED:
                    return "Disconnected";
                case BluetoothProfile.STATE_CONNECTING:
                    return "Connecting";
                case BluetoothProfile.STATE_DISCONNECTING:
                    return "Disconnecting";
                default:
                    return String.valueOf(status);
            }
        }

    }
}


//---------------------------------------------------------------

public class Result extends AppCompatActivity {

 Button Scan, Connect, Start, Stop;
 private final String TAG = "Result";
 private final String DEVICE_NAME = "SML BIO";
 private final String BLUETOOTH_ADRESS = "54:6C:0E:AC:D5:DF";
 private static final int PERMISSION_REQUEST_COARSE_LOCATION = 456;
 private static final long SCAN_PERIOD = 10000;//60000
 private BluetoothGatt mConnectedGatt;
 private BluetoothDevice mConnectedDevice;
 private BluetoothAdapter mBluetoothAdapter;
 PackageManager mManager;
 BluetoothLeScanner mBluetoothLeScanner;
 public BluetoothDevice mDevice;
 public SparseArray<BluetoothDevice> mDevices;
 ArrayList<String> nameDevices;
 Handler mHandler;
 boolean mScanning;
 boolean mScanInprogress;
 private ScanCallback mScanCallback;
 private Sensor mSensor;
 //...

 protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);
		
//...

Scan = findViewById(R.id.scan);
Connect = findViewById(R.id.connect);
Start = findViewById(R.id.start);
Stop  = findViewById(R.id.stop);

BluetoothManager manager = (BluetoothManager) this.getSystemService(this.BLUETOOTH_SERVICE);
mBluetoothAdapter = manager.getAdapter();
mManager = this.getPackageManager();
mBluetoothLeScanner = mBluetoothAdapter.getBluetoothLeScanner();
mDevices = new SparseArray<>();
nameDevices = new ArrayList<String>();
mHandler = new Handler();
mScanning = false;
mScanInprogress = false;        

//...

Start.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mSensor.startAcquisition();
            }});
        Stop.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {    
               mSensor.stopAcquisition();       
            }});

      Connect.setOnClickListener(new View.OnClickListener() {
          public void onClick(View v) {
              
              if(mDevice!=null) {           
                  mSensor = new Sensor(mDevice, getApplicationContext(), mainHandler);
              }
              else{
                  Toast.makeText(getApplicationContext(), "SML BIO Not found", Toast.LENGTH_LONG).show();    
              }
          }});
        Scan.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                if(mBluetoothAdapter.isEnabled()){
                    scanLeDevice(mBluetoothAdapter.isEnabled());
                }
                else{
                    Toast.makeText(getApplicationContext(), "Bluetooth is not Enabled", Toast.LENGTH_LONG).show();
                }
            }});

        //-----------------PERMISSION REQUEST---------------------------------------------
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(new String[]{Manifest.permission.ACCESS_COARSE_LOCATION}, PERMISSION_REQUEST_COARSE_LOCATION);
            requestPermissions(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_WRITE_STORAGE);
            requestPermissions(new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, REQUEST_WRITE_STORAGE);
        }
        //----------------------------------------------------------------------------------
        mScanCallback = new ScanCallback() {

            public void onScanResult(int callbackType, ScanResult result) {
                super.onScanResult(callbackType, result);

                if (result!=null){
                     nameDevices.add(result.getDevice().getName());
                    if(BLUETOOTH_ADRESS.equals(result.getDevice().getAddress())){
                        Toast.makeText(getApplicationContext(),"SML BIO Found", Toast.LENGTH_LONG).show();
                        mDevice = result.getDevice();
                        //-------CREATE SENSOR DEVICE-----------------------
                        mSensor = new Sensor( mDevice, getApplicationContext(), mainHandler);
                     
                        //--------------------------------------------------
                         mDevices.add(result.getDevice());
                    }

            }
            //-----------------------------------------------------
        }};



}

private void scanLeDevice(final boolean enable){
        if (enable) {
            // Stops scanning after a pre-defined scan period.
            mHandler.postDelayed(new Runnable() {
                @Override
                public void run() {
                    mScanning = false;
                    stopScanLe();
                }
            }, SCAN_PERIOD);
            Toast.makeText(getApplicationContext(), "Discovering SML Devices...", Toast.LENGTH_LONG).show();
            mScanning = true;
            startScanLe();
        } else {
            mScanning = false;
            stopScanLe();
            Toast.makeText(getApplicationContext(), "Bluetooth not Disabled", Toast.LENGTH_LONG).show();
        }

}

    public void startScanLe(){

        mBluetoothLeScanner.startScan((ScanCallback) mScanCallback);
        Toast.makeText(getApplicationContext(), "Scanning", Toast.LENGTH_LONG).show();
        mScanInprogress = true;
    }
    public void stopScanLe() {
        mBluetoothLeScanner.stopScan((ScanCallback) mScanCallback);
        Toast.makeText(getApplicationContext(), "Stop Scanning", Toast.LENGTH_LONG).show();
        mScanInprogress = false;
    }



}