&#x20;                EXISTING CCTV

&#x20;                      │

&#x20;                      ▼

&#x20;               RTSP / IP Stream

&#x20;                      │

&#x20;                      ▼

&#x20;             ┌──────────────────┐

&#x20;             │ OpenCV + FFmpeg  │

&#x20;             └────────┬─────────┘

&#x20;                      ▼

&#x20;             ┌──────────────────┐

&#x20;             │ Quality Analyzer │

&#x20;             └────────┬─────────┘

&#x20;                      │

&#x20;             ┌────────┴────────┐

&#x20;             │                 │

&#x20;         Good Quality       Poor Quality

&#x20;             │                 │

&#x20;             │        ┌────────┴─────────┐

&#x20;             │        │                  │

&#x20;             │     Low-light          Low-res

&#x20;             │        │                  │

&#x20;             │    Zero-DCE++        Real-ESRGAN

&#x20;             │        │                  │

&#x20;             │        └────────┬─────────┘

&#x20;             │                 │

&#x20;             └────────┬────────┘

&#x20;                      ▼

&#x20;               Enhanced Video

&#x20;                      │

&#x20;                      ▼

&#x20;                 YOLO Detection

&#x20;                      │

&#x20;                      ▼

&#x20;                 ByteTrack

&#x20;                      │

&#x20;       ┌──────────────┼──────────────┐

&#x20;       ▼              ▼              ▼

&#x20;   Intrusion         ANPR          Face

&#x20;   Detection       Detection     Recognition

&#x20;       │              │              │

&#x20;       └──────────────┼──────────────┘

&#x20;                      ▼

&#x20;                 Risk Engine

&#x20;                      ▼

&#x20;                   Alerts









**FOR THE CCTV VIDEO ENHANCEMENTS** :





&#x20;               CCTV VIDEO

&#x20;                   ↓

&#x20;            Quality Analyzer

&#x20;                   ↓

&#x20;         ┌─────────┴─────────┐

&#x20;         ↓                   ↓

&#x20;     Normal                Poor

&#x20;         ↓                   ↓

&#x20;    No processing      Detect problem

&#x20;                             ↓

&#x20;                ┌────────────┼────────────┐

&#x20;                ↓            ↓            ↓

&#x20;             Low-res      Low-light     Blur

&#x20;                ↓            ↓            ↓

&#x20;          BasicVSR++      Zero-DCE++    RVRT/

&#x20;          /Real-ESRGAN                  Restormer

&#x20;                └────────────┼────────────┘

&#x20;                             ↓

&#x20;                      Enhanced Video











**low-light CCTV**   :    https://github.com/Li-Chongyi/Zero-DCE?utm\_source=chatgpt.com



**Blur**   :   https://github.com/JingyunLiang/RVRT?utm\_source=chatgpt.com



**Low Resolution**   :   https://github.com/xinntao/Real-ESRGAN?utm\_source=chatgpt.com

&#x09;	     https://github.com/ckkelvinchan/BasicVSR\_PlusPlus?utm\_source=chatgpt.com







**Object detection** : https://docs.ultralytics.com/models/yolo26/?utm\_source=chatgpt.com



It should detect:



Person

Car

Truck

Bus

Motorcycle

Bicycle

Other relevant objects







**Object Tracking** :  https://github.com/ifzhang/ByteTrack?utm\_source=chatgpt.com





**Store the trajectory** : Redis + PostgreSQL



&#x09;		Redis is good for fast, temporary tracking data

&#x09;		PostgreSQL is better for permanent information



Example:



Person #17



Time       Position

09:31:01   (120, 250)

09:31:02   (130, 255)

09:31:03   (140, 265)

09:31:04   (150, 280)



You now know:



where they came from

where they're going

how long they've been there

how fast they're moving







| **Task**                 | **Technology**                      |

| -------------------- | ------------------------------- |

| Person detection     |   YOLO                          |

| Object ID            |   ByteTrack                     |

| Position             | Bounding-box bottom-center      |

| Virtual fence        |   Polygon coordinates           |

| Inside/outside check |   OpenCV  pointPolygonTest()    |







**Face Recognition** :: https://github.com/deepinsight/insightface?utm\_source=chatgpt.com

Face Detection	        SCRFD

Face Recognition	ArcFace

