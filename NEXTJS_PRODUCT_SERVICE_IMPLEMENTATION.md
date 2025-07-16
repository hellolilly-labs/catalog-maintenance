# Product Service Implementation for NextJS/TypeScript

This document provides a complete implementation guide for adding product service capabilities to your NextJS/TypeScript application. The goal is to enrich LiveKit agent metadata with current product information and account-specific welcome messages, eliminating network calls in the voice agent entrypoint.

## Overview

The implementation includes:
1. Product data model matching the Python implementation
2. Redis-backed product manager for efficient access
3. GCS fallback for when Redis is unavailable
4. Smart URL-to-product extraction
5. Account configuration with welcome messages

## 1. Product Data Model

Create `types/product.ts`:

```typescript
export interface Product {
  account: string;
  product_id: string;
  descriptor: string;
  name: string;
  price?: string;
  sale_price?: string;
  url: string;
  image_url?: string;
  availability?: string;
  brand?: string;
  category?: string;
  tags?: string[];
  metadata?: Record<string, any>;
  features?: string[];
  specifications?: Record<string, string>;
  variants?: ProductVariant[];
  created_at?: string;
  updated_at?: string;
}

export interface ProductVariant {
  variant_id: string;
  name: string;
  price?: string;
  availability?: string;
  attributes?: Record<string, string>;
}

export interface AccountConfig {
  account: string;
  domain: string;
  company_name: string;
  default_greeting?: string;
  url_patterns?: {
    product_id_pattern?: string;
    product_url_pattern?: string;
  };
  voice_settings?: {
    voice_id?: string;
    voice_provider?: string;
  };
}
```

## 2. Redis Product Manager

Create `services/redisProductManager.ts`:

```typescript
import { Redis } from 'ioredis';
import { Product, AccountConfig } from '../types/product';

export class RedisProductManager {
  private redis: Redis;
  private account: string;
  private productCountCache: number | null = null;
  private accountConfigCache: AccountConfig | null = null;
  private configCacheTime: number = 0;
  private readonly CONFIG_CACHE_TTL = 300000; // 5 minutes

  constructor(account: string, redisClient: Redis) {
    this.account = account;
    this.redis = redisClient;
  }

  /**
   * Find product by ID from Redis
   */
  async findProductById(productId: string): Promise<Product | null> {
    try {
      const key = `products:${this.account}:${productId}`;
      const data = await this.redis.get(key);
      
      if (!data) {
        return null;
      }

      return JSON.parse(data) as Product;
    } catch (error) {
      console.error(`Error fetching product ${productId}:`, error);
      return null;
    }
  }

  /**
   * Find product by URL using smart extraction
   */
  async findProductFromUrl(url: string): Promise<Product | null> {
    try {
      // First try smart extraction using URL patterns
      const productId = await this.extractProductIdFromUrl(url);
      if (productId) {
        const product = await this.findProductById(productId);
        if (product) {
          return product;
        }
      }

      // Fallback to URL index lookup
      const urlKey = this.normalizeUrl(url);
      const indexKey = `url_index:${this.account}:${urlKey}`;
      const productId2 = await this.redis.get(indexKey);
      
      if (productId2) {
        return await this.findProductById(productId2);
      }

      return null;
    } catch (error) {
      console.error(`Error finding product by URL ${url}:`, error);
      return null;
    }
  }

  /**
   * Extract product ID from URL using account-specific patterns
   */
  private async extractProductIdFromUrl(url: string): Promise<string | null> {
    try {
      const config = await this.getAccountConfig();
      if (!config?.url_patterns?.product_id_pattern) {
        return null;
      }

      const pattern = new RegExp(config.url_patterns.product_id_pattern);
      const match = url.match(pattern);
      
      if (match && match[1]) {
        return match[1];
      }

      return null;
    } catch (error) {
      console.error('Error extracting product ID from URL:', error);
      return null;
    }
  }

  /**
   * Get account configuration with caching
   */
  async getAccountConfig(): Promise<AccountConfig | null> {
    try {
      // Check cache
      const now = Date.now();
      if (this.accountConfigCache && (now - this.configCacheTime) < this.CONFIG_CACHE_TTL) {
        return this.accountConfigCache;
      }

      // Fetch from Redis
      const key = `account_config:${this.account}`;
      const data = await this.redis.get(key);
      
      if (!data) {
        return null;
      }

      this.accountConfigCache = JSON.parse(data) as AccountConfig;
      this.configCacheTime = now;
      
      return this.accountConfigCache;
    } catch (error) {
      console.error(`Error fetching account config for ${this.account}:`, error);
      return null;
    }
  }

  /**
   * Get default greeting for account
   */
  async getDefaultGreeting(): Promise<string> {
    const config = await this.getAccountConfig();
    return config?.default_greeting || "Hello! How can I help you today?";
  }

  /**
   * Get total product count
   */
  async getProductCount(): Promise<number> {
    if (this.productCountCache !== null) {
      return this.productCountCache;
    }

    try {
      const key = `product_count:${this.account}`;
      const count = await this.redis.get(key);
      this.productCountCache = count ? parseInt(count, 10) : 0;
      return this.productCountCache;
    } catch (error) {
      console.error('Error fetching product count:', error);
      return 0;
    }
  }

  /**
   * Check if products are available in Redis
   */
  async hasProducts(): Promise<boolean> {
    const count = await this.getProductCount();
    return count > 0;
  }

  /**
   * Normalize URL for consistent lookups
   */
  private normalizeUrl(url: string): string {
    try {
      const urlObj = new URL(url);
      // Remove protocol, www, trailing slashes, and query params
      let normalized = urlObj.hostname + urlObj.pathname;
      normalized = normalized.replace(/^www\./, '');
      normalized = normalized.replace(/\/$/, '');
      normalized = normalized.toLowerCase();
      return normalized;
    } catch {
      // Fallback for invalid URLs
      return url.toLowerCase().replace(/^https?:\/\//, '').replace(/^www\./, '').replace(/\/$/, '');
    }
  }
}
```

## 3. GCS Fallback Service

Create `services/gcsProductService.ts`:

```typescript
import { Storage } from '@google-cloud/storage';
import { Product, AccountConfig } from '../types/product';

export class GCSProductService {
  private storage: Storage;
  private bucketName: string;
  private productsCache: Map<string, Product[]> = new Map();
  private configCache: Map<string, AccountConfig> = new Map();

  constructor(bucketName: string = 'liddy-account-documents') {
    this.storage = new Storage();
    this.bucketName = bucketName;
  }

  /**
   * Load products from GCS bucket
   */
  async loadProducts(account: string): Promise<Product[]> {
    // Check cache first
    if (this.productsCache.has(account)) {
      return this.productsCache.get(account)!;
    }

    try {
      const file = this.storage.bucket(this.bucketName).file(`accounts/${account}/products.json`);
      const [exists] = await file.exists();
      
      if (!exists) {
        console.warn(`No products file found for ${account}`);
        return [];
      }

      const [contents] = await file.download();
      const products = JSON.parse(contents.toString()) as Product[];
      
      // Cache for future use
      this.productsCache.set(account, products);
      
      return products;
    } catch (error) {
      console.error(`Error loading products from GCS for ${account}:`, error);
      return [];
    }
  }

  /**
   * Load account configuration from GCS
   */
  async loadAccountConfig(account: string): Promise<AccountConfig | null> {
    // Check cache first
    if (this.configCache.has(account)) {
      return this.configCache.get(account)!;
    }

    try {
      const file = this.storage.bucket(this.bucketName).file(`accounts/${account}/account.json`);
      const [exists] = await file.exists();
      
      if (!exists) {
        console.warn(`No account config found for ${account}`);
        return null;
      }

      const [contents] = await file.download();
      const config = JSON.parse(contents.toString()) as AccountConfig;
      
      // Cache for future use
      this.configCache.set(account, config);
      
      return config;
    } catch (error) {
      console.error(`Error loading account config from GCS for ${account}:`, error);
      return null;
    }
  }

  /**
   * Find product by URL from GCS data
   */
  async findProductByUrl(account: string, url: string): Promise<Product | null> {
    const products = await this.loadProducts(account);
    const normalizedUrl = this.normalizeUrl(url);
    
    return products.find(p => this.normalizeUrl(p.url) === normalizedUrl) || null;
  }

  private normalizeUrl(url: string): string {
    try {
      const urlObj = new URL(url);
      let normalized = urlObj.hostname + urlObj.pathname;
      normalized = normalized.replace(/^www\./, '');
      normalized = normalized.replace(/\/$/, '');
      normalized = normalized.toLowerCase();
      return normalized;
    } catch {
      return url.toLowerCase().replace(/^https?:\/\//, '').replace(/^www\./, '').replace(/\/$/, '');
    }
  }
}
```

## 4. Unified Product Service

Create `services/productService.ts`:

```typescript
import { Redis } from 'ioredis';
import { Product, AccountConfig } from '../types/product';
import { RedisProductManager } from './redisProductManager';
import { GCSProductService } from './gcsProductService';

export class ProductService {
  private redisManager: RedisProductManager;
  private gcsService: GCSProductService;
  private account: string;

  constructor(account: string, redisClient: Redis) {
    this.account = account;
    this.redisManager = new RedisProductManager(account, redisClient);
    this.gcsService = new GCSProductService();
  }

  /**
   * Find product by current URL with Redis + GCS fallback
   */
  async findProductByUrl(url: string): Promise<Product | null> {
    try {
      // Try Redis first
      const hasRedisProducts = await this.redisManager.hasProducts();
      if (hasRedisProducts) {
        const product = await this.redisManager.findProductFromUrl(url);
        if (product) {
          return product;
        }
      }

      // Fallback to GCS
      console.log(`Redis lookup failed for ${url}, falling back to GCS`);
      return await this.gcsService.findProductByUrl(this.account, url);
    } catch (error) {
      console.error('Error finding product by URL:', error);
      return null;
    }
  }

  /**
   * Get account configuration with fallback
   */
  async getAccountConfig(): Promise<AccountConfig | null> {
    try {
      // Try Redis first
      const config = await this.redisManager.getAccountConfig();
      if (config) {
        return config;
      }

      // Fallback to GCS
      return await this.gcsService.loadAccountConfig(this.account);
    } catch (error) {
      console.error('Error getting account config:', error);
      return null;
    }
  }

  /**
   * Get default greeting for account
   */
  async getDefaultGreeting(): Promise<string> {
    const config = await this.getAccountConfig();
    return config?.default_greeting || "Hello! How can I help you today?";
  }

  /**
   * Prepare metadata for LiveKit agent
   */
  async prepareLiveKitMetadata(userId: string, currentUrl?: string): Promise<Record<string, any>> {
    const metadata: Record<string, any> = {
      userId,
      account: this.account,
      timestamp: new Date().toISOString()
    };

    // Add current URL if provided
    if (currentUrl) {
      metadata.currentUrl = currentUrl;

      // Try to find and include current product
      const product = await this.findProductByUrl(currentUrl);
      if (product) {
        metadata.currentProduct = {
          product_id: product.product_id,
          name: product.name,
          price: product.price,
          category: product.category,
          url: product.url
        };
      }
    }

    // Include welcome message
    metadata.welcomeMessage = await this.getDefaultGreeting();

    // Include account config details
    const config = await this.getAccountConfig();
    if (config) {
      metadata.accountConfig = {
        company_name: config.company_name,
        voice_settings: config.voice_settings
      };
    }

    return metadata;
  }
}
```

## 5. Integration Example

Here's how to integrate this into your NextJS application:

### API Route Example

Create `pages/api/livekit/prepare-metadata.ts`:

```typescript
import { NextApiRequest, NextApiResponse } from 'next';
import Redis from 'ioredis';
import { ProductService } from '../../../services/productService';

// Initialize Redis client (reuse existing if available)
const redis = new Redis({
  host: process.env.REDIS_HOST || 'localhost',
  port: parseInt(process.env.REDIS_PORT || '6379'),
  password: process.env.REDIS_PASSWORD,
});

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { userId, account, currentUrl } = req.body;

  if (!userId || !account) {
    return res.status(400).json({ error: 'Missing required parameters' });
  }

  try {
    const productService = new ProductService(account, redis);
    const metadata = await productService.prepareLiveKitMetadata(userId, currentUrl);
    
    res.status(200).json({ metadata });
  } catch (error) {
    console.error('Error preparing metadata:', error);
    res.status(500).json({ error: 'Failed to prepare metadata' });
  }
}
```

### Usage in LiveKit Agent Creation

```typescript
import { createLiveKitAgent } from './livekit-client';

async function startVoiceAgent(userId: string, account: string, currentUrl?: string) {
  // Prepare metadata with product info
  const response = await fetch('/api/livekit/prepare-metadata', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ userId, account, currentUrl })
  });

  const { metadata } = await response.json();

  // Create LiveKit agent with enriched metadata
  const agent = await createLiveKitAgent({
    roomName: `${account}_${generateRoomId()}`,
    participantMetadata: JSON.stringify(metadata),
    // ... other LiveKit options
  });

  return agent;
}
```

## 6. Environment Variables

Add these to your `.env`:

```bash
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Google Cloud Storage
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GCS_BUCKET_NAME=liddy-account-documents
```

## 7. Benefits

1. **Zero Network Calls in Voice Agent**: All product data and welcome messages are included in metadata
2. **Efficient Caching**: Redis provides fast access with GCS fallback
3. **Smart URL Parsing**: Account-specific patterns for product extraction
4. **Type Safety**: Full TypeScript support matching Python models
5. **Scalable**: Redis shared cache works across multiple containers

## 8. Testing

Create `services/__tests__/productService.test.ts`:

```typescript
import { ProductService } from '../productService';
import Redis from 'ioredis-mock';

describe('ProductService', () => {
  let productService: ProductService;
  let redis: Redis;

  beforeEach(() => {
    redis = new Redis();
    productService = new ProductService('specialized.com', redis);
  });

  test('should find product by URL', async () => {
    // Mock Redis data
    await redis.set('products:specialized.com:12345', JSON.stringify({
      product_id: '12345',
      name: 'Test Product',
      url: 'https://specialized.com/products/12345'
    }));
    await redis.set('url_index:specialized.com:specialized.com/products/12345', '12345');

    const product = await productService.findProductByUrl('https://specialized.com/products/12345');
    expect(product).toBeTruthy();
    expect(product?.product_id).toBe('12345');
  });

  test('should prepare LiveKit metadata', async () => {
    const metadata = await productService.prepareLiveKitMetadata('user123', 'https://specialized.com/products/12345');
    
    expect(metadata).toHaveProperty('userId', 'user123');
    expect(metadata).toHaveProperty('account', 'specialized.com');
    expect(metadata).toHaveProperty('welcomeMessage');
    expect(metadata).toHaveProperty('currentUrl');
  });
});
```

## Implementation Notes

1. The Redis keys match the Python implementation exactly for compatibility
2. URL normalization follows the same logic as Python
3. The Product interface matches all fields from the Python dataclass
4. Account config includes welcome messages and voice settings
5. GCS fallback ensures reliability when Redis is unavailable
6. Metadata preparation includes all necessary context for the voice agent

This implementation will eliminate network calls in your voice agent's entrypoint by providing all necessary context upfront through LiveKit metadata.