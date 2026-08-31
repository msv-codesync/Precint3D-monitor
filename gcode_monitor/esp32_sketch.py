"""Arduino sketch handed to the operator so an ESP32-CAM can feed the console.

The board only has to do one thing: answer GET /capture with a JPEG. Keeping the
firmware that small means the console treats it exactly like a phone camera app.
"""

from __future__ import annotations

BOARDS = {
    "AI-Thinker ESP32-CAM": "CAMERA_MODEL_AI_THINKER",
    "ESP32-S3 XIAO Sense": "CAMERA_MODEL_XIAO_ESP32S3",
    "M5Stack Timer Camera": "CAMERA_MODEL_M5STACK_PSRAM",
}

_TEMPLATE = '''// Precint3D camera node - answers GET /capture with one JPEG.
// Board: {board_label}
// Arduino IDE: Tools > Board > ESP32 Arduino > {board_label}
//              Tools > Partition Scheme > Huge APP
// After upload, open Serial Monitor at 115200 and copy the IP it prints.

#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>

#define {board_define}
#include "camera_pins.h"

const char* WIFI_SSID = "{ssid}";
const char* WIFI_PASS = "{password}";

WebServer server(80);

static void handleCapture() {{
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {{
    server.send(503, "text/plain", "capture failed");
    return;
  }}
  server.setContentLength(fb->len);
  server.send(200, "image/jpeg", "");
  server.sendContent(reinterpret_cast<const char*>(fb->buf), fb->len);
  esp_camera_fb_return(fb);
}}

static bool startCamera() {{
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;   config.pin_pclk  = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM; config.pin_href  = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn  = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode    = CAMERA_GRAB_LATEST;
  config.fb_location  = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
  // 800x600 keeps the print bed readable without stalling the YOLO step.
  config.frame_size = psramFound() ? FRAMESIZE_SVGA : FRAMESIZE_VGA;
  config.jpeg_quality = psramFound() ? 12 : 15;
  config.fb_count     = psramFound() ? 2 : 1;
  return esp_camera_init(&config) == ESP_OK;
}}

void setup() {{
  Serial.begin(115200);
  Serial.setDebugOutput(false);

  if (!startCamera()) {{
    Serial.println("Camera init failed. Check the ribbon cable and power.");
    return;
  }}

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Joining ");
  Serial.println(WIFI_SSID);
  while (WiFi.status() != WL_CONNECTED) {{
    delay(400);
    Serial.print(".");
  }}

  server.on("/capture", HTTP_GET, handleCapture);
  server.on("/", HTTP_GET, []() {{
    server.send(200, "text/plain", "Precint3D camera node. Use /capture");
  }});
  server.begin();

  Serial.println();
  Serial.println("=====================================");
  Serial.print("Paste this into Precint3D:  ");
  Serial.println(WiFi.localIP());
  Serial.println("=====================================");
}}

void loop() {{
  server.handleClient();
}}
'''


def sketch_for(ssid: str, password: str, board: str = "AI-Thinker ESP32-CAM") -> str:
    """Fill the sketch with the operator's network so nothing has to be edited by hand."""
    return _TEMPLATE.format(
        board_label=board,
        board_define=BOARDS.get(board, BOARDS["AI-Thinker ESP32-CAM"]),
        ssid=(ssid or "YOUR_WIFI_NAME").replace('"', ""),
        password=(password or "YOUR_WIFI_PASSWORD").replace('"', ""),
    )
