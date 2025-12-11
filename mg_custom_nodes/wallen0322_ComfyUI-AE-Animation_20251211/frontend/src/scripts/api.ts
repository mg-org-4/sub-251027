class ComfyAPI {
  apiURL(path: string): string {
    const base = window.location.origin
    return `${base}${path}`
  }

  async fetchApi(path: string, options?: RequestInit): Promise<Response> {
    return fetch(this.apiURL(path), options)
  }
}

export const api = new ComfyAPI()
