import { CapacitorConfig } from '@capacitor/cli'

const config: CapacitorConfig = {
  appId: 'com.niquewrld.cocoadisease',
  appName: 'Cocoa Disease Detector',
  webDir: 'dist',
  server: {
    androidScheme: 'https',
  },
  plugins: {
    Camera: {
      allowEditing: false,
      resultType: 'uri',
      source: 'CAMERA',
    },
  },
}

export default config
