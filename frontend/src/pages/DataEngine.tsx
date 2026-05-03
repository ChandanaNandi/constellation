import { useState, useEffect } from 'react'
import { Upload, RefreshCw, Filter, ChevronLeft, ChevronRight } from 'lucide-react'
import { api } from '../api/client'
import ImageGrid from '../components/ImageGrid'
import ImageModal from '../components/ImageModal'

interface Image {
  id: number
  filename: string
  file_path: string
  width: number
  height: number
  status: string
  dataset: string
  created_at: string
}

interface ImageListResponse {
  images: Image[]
  total: number
  limit: number
  offset: number
}

export default function DataEngine() {
  const [images, setImages] = useState<Image[]>([])
  const [total, setTotal] = useState(0)
  const [offset, setOffset] = useState(0)
  const [loading, setLoading] = useState(true)
  const [selectedImage, setSelectedImage] = useState<any>(null)
  const [isLabeling, setIsLabeling] = useState<number | null>(null)
  const [statusFilter, setStatusFilter] = useState('')
  const [datasetFilter, setDatasetFilter] = useState('')

  const limit = 20

  const fetchImages = async () => {
    setLoading(true)
    try {
      const data: ImageListResponse = await api.getImages({
        limit,
        offset,
        status: statusFilter || undefined,
        dataset: datasetFilter || undefined,
      })
      setImages(data.images)
      setTotal(data.total)
    } catch (err) {
      console.error('Failed to fetch images:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchImages()
  }, [offset, statusFilter, datasetFilter])

  const handleImageClick = async (image: Image) => {
    try {
      const detail = await api.getImage(image.id)
      setSelectedImage(detail)
    } catch (err) {
      console.error('Failed to fetch image detail:', err)
    }
  }

  const handleAutoLabel = async (imageId: number) => {
    setIsLabeling(imageId)
    try {
      await api.autoLabel(imageId)
      // Refresh the image detail if modal is open
      if (selectedImage?.image?.id === imageId) {
        const detail = await api.getImage(imageId)
        setSelectedImage(detail)
      }
      // Refresh grid
      fetchImages()
    } catch (err) {
      console.error('Auto-label failed:', err)
      alert('Auto-labeling failed. Make sure ML dependencies are installed in the backend.')
    } finally {
      setIsLabeling(null)
    }
  }

  const totalPages = Math.ceil(total / limit)
  const currentPage = Math.floor(offset / limit) + 1

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">Data Engine (BDD100K Labeling Ops)</h1>
          <p className="mt-1 text-slate-600">
            {total} images in database
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={fetchImages}
            className="flex items-center gap-2 px-4 py-2 text-slate-700 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>

          <button className="flex items-center gap-2 px-4 py-2 text-white bg-constellation-600 rounded-lg hover:bg-constellation-700 transition-colors">
            <Upload className="h-4 w-4" />
            Upload
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-4 mb-6 p-4 bg-white rounded-lg border border-slate-200">
        <Filter className="h-5 w-5 text-slate-400" />
        <select
          value={statusFilter}
          onChange={(e) => {
            setOffset(0)
            setStatusFilter(e.target.value)
          }}
          className="px-3 py-1.5 border border-slate-200 rounded-lg text-sm"
        >
          <option value="">All Status</option>
          <option value="pending">Pending</option>
          <option value="labeled">Labeled</option>
          <option value="reviewed">Reviewed</option>
        </select>
        <select
          value={datasetFilter}
          onChange={(e) => {
            setOffset(0)
            setDatasetFilter(e.target.value)
          }}
          className="px-3 py-1.5 border border-slate-200 rounded-lg text-sm"
        >
          <option value="">All Datasets</option>
          <option value="cityscapes">Cityscapes</option>
          <option value="bdd100k">BDD100K</option>
          <option value="uploaded">Uploaded</option>
        </select>
      </div>

      <div className="mb-6 rounded-lg border border-blue-200 bg-blue-50 px-4 py-3 text-sm text-blue-900">
        This page is for labeling operations on database images (primarily BDD100K). HydraNet V2 model demos are
        Cityscapes-aligned and should be validated in the dedicated inference demo flow.
      </div>

      {/* Image Grid */}
      {loading ? (
        <div className="flex items-center justify-center h-64">
          <RefreshCw className="h-8 w-8 text-slate-400 animate-spin" />
        </div>
      ) : images.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-64 bg-white rounded-lg border border-slate-200">
          <p className="text-slate-500 mb-4">No images found</p>
          <button className="flex items-center gap-2 px-4 py-2 text-white bg-constellation-600 rounded-lg hover:bg-constellation-700 transition-colors">
            <Upload className="h-4 w-4" />
            Upload Images
          </button>
        </div>
      ) : (
        <ImageGrid
          images={images}
          onImageClick={handleImageClick}
          onAutoLabel={handleAutoLabel}
          isLabeling={isLabeling}
        />
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-4 mt-6">
          <button
            onClick={() => setOffset(Math.max(0, offset - limit))}
            disabled={offset === 0}
            className="flex items-center gap-1 px-3 py-2 text-slate-600 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ChevronLeft className="h-4 w-4" />
            Previous
          </button>

          <span className="text-sm text-slate-600">
            Page {currentPage} of {totalPages}
          </span>

          <button
            onClick={() => setOffset(offset + limit)}
            disabled={offset + limit >= total}
            className="flex items-center gap-1 px-3 py-2 text-slate-600 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Next
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      )}

      {/* Image Modal */}
      <ImageModal
        imageDetail={selectedImage}
        onClose={() => setSelectedImage(null)}
      />
    </div>
  )
}
