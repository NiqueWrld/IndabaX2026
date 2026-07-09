import { useEffect, useRef, useState } from 'react'
import { Camera, CameraResultType, CameraSource } from '@capacitor/camera'
import * as ort from 'onnxruntime-web'

const CLASS_NAMES = ['anthracnose', 'cssvd', 'healthy'] as const
type ClassName = (typeof CLASS_NAMES)[number]

const EXAMPLES = [
  { label: 'Anthracnose', src: '/examples/example_anthracnose.jpg' },
  { label: 'CSSVD', src: '/examples/example_cssvd.jpg' },
  { label: 'Healthy', src: '/examples/example_healthy.jpg' },
] as const

interface Detection {
  className: ClassName
  confidence: number
  x1: number
  y1: number
  x2: number
  y2: number
}

const MODEL_SIZE = 1024
const CONF_THRESHOLD = 0.20
const IOU_THRESHOLD = 0.45

function App() {
  const [image, setImage] = useState<string | null>(null)
  const [results, setResults] = useState<Detection[]>([])
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState<string>('')
  const [error, setError] = useState<string>('')
  const sessionRef = useRef<ort.InferenceSession | null>(null)

  useEffect(() => {
    loadModel()
  }, [])

  const loadModel = async () => {
    try {
      setStatus('Loading model...')
      sessionRef.current = await ort.InferenceSession.create('/cocoa-disease.onnx', {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      })
      setStatus('Model ready')
    } catch (err) {
      setError(`Failed to load model: ${err}`)
      setStatus('')
    }
  }

  const takePhoto = async () => {
    try {
      setError('')
      const photo = await Camera.getPhoto({
        quality: 90,
        allowEditing: false,
        resultType: CameraResultType.Uri,
        source: CameraSource.Prompt,
      })
      if (photo.webPath) {
        setImage(photo.webPath)
        setResults([])
      }
    } catch (err) {
      setError(`Camera error: ${err}`)
    }
  }

  const runInference = async () => {
    if (!image || !sessionRef.current) {
      setError('Please capture an image and wait for the model to load.')
      return
    }

    try {
      setLoading(true)
      setError('')
      setStatus('Processing image...')

      const img = await loadImage(image)
      const { input, scale, padX, padY } = letterbox(img, MODEL_SIZE)
      const tensor = new ort.Tensor('float32', input, [1, 3, MODEL_SIZE, MODEL_SIZE])

      const feeds: Record<string, ort.Tensor> = { images: tensor }
      const output = await sessionRef.current.run(feeds)
      const outputName = sessionRef.current.outputNames[0]
      const predictions = output[outputName].data as Float32Array

      const detections = postprocess(predictions, scale, padX, padY, img.width, img.height)
      setResults(detections)
      setStatus(`Found ${detections.length} detection(s)`)
    } catch (err) {
      setError(`Inference error: ${err}`)
      setStatus('')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Cocoa Disease Detector</h1>
        <p>Take a photo of a cocoa leaf to detect diseases</p>
      </header>

      {status && <div className={`status ${loading ? 'loading' : ''}`}>{status}</div>}
      {error && <div className="status error">{error}</div>}

      <div className="card">
        {image ? (
          <img src={image} alt="Captured cocoa leaf" className="image-preview" />
        ) : (
          <div className="placeholder">No image captured</div>
        )}
        <div className="button-group">
          <button className="btn-primary" onClick={takePhoto} disabled={loading}>
            {image ? 'Retake' : 'Capture'}
          </button>
          <button className="btn-secondary" onClick={runInference} disabled={!image || loading}>
            {loading ? 'Analyzing...' : 'Analyze'}
          </button>
        </div>

        <div className="examples">
          <p className="examples-label">Or try an example</p>
          <div className="example-thumbs">
            {EXAMPLES.map((ex) => (
              <button
                key={ex.src}
                className="example-btn"
                onClick={() => {
                  setImage(ex.src)
                  setResults([])
                  setError('')
                }}
                disabled={loading}
              >
                <img src={ex.src} alt={ex.label} />
                <span>{ex.label}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {results.length > 0 && (
        <div className="card results">
          <h2>Results</h2>
          {results.map((det, i) => (
            <div key={i} className={`result-item ${det.className}`}>
              <span className="result-class">{det.className}</span>
              <span className="result-conf">{(det.confidence * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      )}

      <div className="footer">
        IndabaX South Africa Cocoa Contamination Hackathon — Bonus mobile demo
      </div>
    </div>
  )
}

async function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = src
  })
}

function letterbox(img: HTMLImageElement, targetSize: number) {
  const canvas = document.createElement('canvas')
  canvas.width = targetSize
  canvas.height = targetSize
  const ctx = canvas.getContext('2d')!

  // Fill gray background (YOLOv8 training padding color)
  ctx.fillStyle = '#808080'
  ctx.fillRect(0, 0, targetSize, targetSize)

  const scale = Math.min(targetSize / img.width, targetSize / img.height)
  const newW = Math.round(img.width * scale)
  const newH = Math.round(img.height * scale)
  const padX = Math.round((targetSize - newW) / 2)
  const padY = Math.round((targetSize - newH) / 2)

  ctx.drawImage(img, padX, padY, newW, newH)

  const imageData = ctx.getImageData(0, 0, targetSize, targetSize)
  const input = new Float32Array(3 * targetSize * targetSize)

  for (let i = 0; i < imageData.data.length / 4; i++) {
    const r = imageData.data[i * 4] / 255
    const g = imageData.data[i * 4 + 1] / 255
    const b = imageData.data[i * 4 + 2] / 255

    input[i] = r
    input[i + targetSize * targetSize] = g
    input[i + 2 * targetSize * targetSize] = b
  }

  return { input, scale, padX, padY }
}

function postprocess(
  data: Float32Array,
  scale: number,
  padX: number,
  padY: number,
  origWidth: number,
  origHeight: number,
): Detection[] {
  // Ultralytics YOLOv8 ONNX output shape: [1, 4 + num_classes, num_anchors]
  const numChannels = 4 + CLASS_NAMES.length
  const numAnchors = data.length / numChannels

  const candidates: Detection[] = []

  for (let i = 0; i < numAnchors; i++) {
    const cx = data[i]
    const cy = data[i + numAnchors]
    const w = data[i + 2 * numAnchors]
    const h = data[i + 3 * numAnchors]

    let bestScore = -Infinity
    let bestClass = 0

    for (let c = 0; c < CLASS_NAMES.length; c++) {
      // Ultralytics ONNX export emits raw class logits; apply sigmoid to get probabilities
      const score = sigmoid(data[i + (4 + c) * numAnchors])
      if (score > bestScore) {
        bestScore = score
        bestClass = c
      }
    }

    if (bestScore < CONF_THRESHOLD) continue

    // Convert center-size to corners and undo letterbox padding
    let x1 = (cx - w / 2 - padX) / scale
    let y1 = (cy - h / 2 - padY) / scale
    let x2 = (cx + w / 2 - padX) / scale
    let y2 = (cy + h / 2 - padY) / scale

    x1 = Math.max(0, Math.min(x1, origWidth))
    y1 = Math.max(0, Math.min(y1, origHeight))
    x2 = Math.max(0, Math.min(x2, origWidth))
    y2 = Math.max(0, Math.min(y2, origHeight))

    candidates.push({
      className: CLASS_NAMES[bestClass],
      confidence: bestScore,
      x1,
      y1,
      x2,
      y2,
    })
  }

  return nms(candidates, IOU_THRESHOLD)
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x))
}

function nms(detections: Detection[], iouThreshold: number): Detection[] {
  detections.sort((a, b) => b.confidence - a.confidence)
  const kept: Detection[] = []

  for (const det of detections) {
    let suppressed = false
    for (const keep of kept) {
      if (keep.className !== det.className) continue
      if (iou(det, keep) > iouThreshold) {
        suppressed = true
        break
      }
    }
    if (!suppressed) kept.push(det)
  }

  return kept
}

function iou(a: Detection, b: Detection): number {
  const x1 = Math.max(a.x1, b.x1)
  const y1 = Math.max(a.y1, b.y1)
  const x2 = Math.min(a.x2, b.x2)
  const y2 = Math.min(a.y2, b.y2)

  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1)
  const union = areaA + areaB - inter

  return union > 0 ? inter / union : 0
}

export default App
