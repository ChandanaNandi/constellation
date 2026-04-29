import axios from 'axios'

const apiClient = axios.create({
  baseURL: import.meta.env.PROD ? '/api' : 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
})

export const api = {
  getHealth: async () => {
    const response = await apiClient.get('/health')
    return response.data
  },

  // Images
  uploadImage: async (file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    const response = await apiClient.post('/api/images/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return response.data
  },

  getImages: async (params?: { limit?: number; offset?: number }) => {
    const response = await apiClient.get('/api/images', { params })
    return response.data
  },

  // Models
  getModels: async () => {
    const response = await apiClient.get('/api/models')
    return response.data
  },

  // Predictions
  runInference: async (imageId: number, modelId: number) => {
    const response = await apiClient.post('/api/predict', { imageId, modelId })
    return response.data
  },

  // Shadow Mode
  getDisagreements: async (params?: { limit?: number; offset?: number }) => {
    const response = await apiClient.get('/api/disagreements', { params })
    return response.data
  },
}

export default apiClient
